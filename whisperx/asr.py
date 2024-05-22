import os
import warnings
from typing import List, Union, Optional, NamedTuple

import ctranslate2
import faster_whisper
import numpy as np
import torch
from transformers import Pipeline
from transformers.pipelines.pt_utils import PipelineIterator

from .audio import N_SAMPLES, SAMPLE_RATE, load_audio, log_mel_spectrogram
from .vad import load_vad_model, merge_chunks
from .types import TranscriptionResult, SingleSegment

def find_numeral_symbol_tokens(tokenizer):
    numeral_symbol_tokens = []
    for i in range(tokenizer.eot):
        token = tokenizer.decode([i]).removeprefix(" ")
        has_numeral_symbol = any(c in "0123456789%$£" for c in token)
        if has_numeral_symbol:
            numeral_symbol_tokens.append(i)
    return numeral_symbol_tokens

class WhisperModel(faster_whisper.WhisperModel):
    '''
    FasterWhisperModel provides batched inference for faster-whisper.
    Currently only works in non-timestamp mode and fixed prompt for all samples in batch.
    '''

    def generate_segment_batched(self, features: np.ndarray, tokenizer: faster_whisper.tokenizer.Tokenizer, options: faster_whisper.transcribe.TranscriptionOptions, encoder_output = None):
        batch_size = features.shape[0]
        all_tokens = []
        prompt_reset_since = 0
        if options.initial_prompt is not None:
            initial_prompt = " " + options.initial_prompt.strip()
            initial_prompt_tokens = tokenizer.encode(initial_prompt)
            all_tokens.extend(initial_prompt_tokens)
        previous_tokens = all_tokens[prompt_reset_since:]
        prompt = self.get_prompt(
            tokenizer,
            previous_tokens,
            without_timestamps=options.without_timestamps,
            prefix=options.prefix,
        )

        encoder_output = self.encode(features)

        max_initial_timestamp_index = int(
            round(options.max_initial_timestamp / self.time_precision)
        )

        result = self.model.generate(
                encoder_output,
                [prompt] * batch_size,
                beam_size=options.beam_size,
                patience=options.patience,
                length_penalty=options.length_penalty,
                max_length=self.max_length,
                suppress_blank=options.suppress_blank,
                suppress_tokens=options.suppress_tokens,
            )

        tokens_batch = [x.sequences_ids[0] for x in result]

        def decode_batch(tokens: List[List[int]]) -> str:
            res = []
            for tk in tokens:
                res.append([token for token in tk if token < tokenizer.eot])
            # text_tokens = [token for token in tokens if token < self.eot]
            return tokenizer.tokenizer.decode_batch(res)

        text = decode_batch(tokens_batch)

        return text

    def encode(self, features: np.ndarray) -> ctranslate2.StorageView:
        # When the model is running on multiple GPUs, the encoder output should be moved
        # to the CPU since we don't know which GPU will handle the next job.
        to_cpu = self.model.device == "cuda" and len(self.model.device_index) > 1
        # unsqueeze if batch size = 1
        if len(features.shape) == 2:
            features = np.expand_dims(features, 0)
        features = faster_whisper.transcribe.get_ctranslate2_storage(features)

        return self.model.encode(features, to_cpu=to_cpu)

class FasterWhisperPipeline(Pipeline):
    """
    Huggingface Pipeline wrapper for FasterWhisperModel.
    """
    # TODO:
    # - add support for timestamp mode
    # - add support for custom inference kwargs

    def __init__(
            self,
            model,
            vad,
            vad_params: dict,
            options : NamedTuple,
            tokenizer=None,
            device: Union[int, str, "torch.device"] = -1,
            framework = "pt",
            language : Optional[str] = None,
            suppress_numerals: bool = False,
            **kwargs
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.options = options
        self.preset_language = None
        self.suppress_numerals = suppress_numerals
        self._batch_size = kwargs.pop("batch_size", None)
        self._num_workers = 1
        self._preprocess_params, self._forward_params, self._postprocess_params = self._sanitize_parameters(**kwargs)
        self.call_count = 0
        self.framework = framework
        if self.framework == "pt":
            if isinstance(device, torch.device):
                self.device = device
            elif isinstance(device, str):
                self.device = torch.device(device)
            elif device < 0:
                self.device = torch.device("cpu")
            else:
                self.device = torch.device(f"cuda:{device}")
        else:
            self.device = device

        super(Pipeline, self).__init__()
        self.vad_model = vad
        self._vad_params = vad_params

    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        if "tokenizer" in kwargs:
            preprocess_kwargs["maybe_arg"] = kwargs["maybe_arg"]
        return preprocess_kwargs, {}, {}

    def preprocess(self, audio):
        audio = audio['inputs']
        model_n_mels = self.model.feat_kwargs.get("feature_size")
        features = log_mel_spectrogram(
            audio,
            n_mels=model_n_mels if model_n_mels is not None else 80,
            padding=N_SAMPLES - audio.shape[0],
        )
        return {'inputs': features}

    def _forward(self, model_inputs):
        outputs = self.model.generate_segment_batched(model_inputs['inputs'], self.tokenizer, self.options)
        return {'text': outputs}

    def postprocess(self, model_outputs):
        return model_outputs

    def get_iterator(
        self, inputs, num_workers: int, batch_size: int, preprocess_params, forward_params, postprocess_params
    ):
        dataset = PipelineIterator(inputs, self.preprocess, preprocess_params)
        if "TOKENIZERS_PARALLELISM" not in os.environ:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
        # TODO hack by collating feature_extractor and image_processor

        def stack(items):
            return {'inputs': torch.stack([x['inputs'] for x in items])}
        dataloader = torch.utils.data.DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, collate_fn=stack)
        model_iterator = PipelineIterator(dataloader, self.forward, forward_params, loader_batch_size=batch_size)
        final_iterator = PipelineIterator(model_iterator, self.postprocess, postprocess_params)
        return final_iterator

    def transcribe(
        self, audio: Union[str, np.ndarray], batch_size=None, num_workers=0, language=None, task=None, chunk_size=30, print_progress = False, combined_progress=False
    ) -> TranscriptionResult:
        
        languages_identified = set()
        if isinstance(audio, str):
            audio = load_audio(audio)

        def data(audio, segments):
            for seg in segments:
                f1 = int(seg['start'] * SAMPLE_RATE)
                f2 = int(seg['end'] * SAMPLE_RATE)
                yield {'inputs': audio[f1:f2]}

        vad_segments = self.vad_model({"waveform": torch.from_numpy(audio).unsqueeze(0), "sample_rate": SAMPLE_RATE})
        vad_segments = merge_chunks(
            vad_segments,
            chunk_size,
            onset=self._vad_params["vad_onset"],
            offset=self._vad_params["vad_offset"],
        )
        if self.tokenizer is None:
            print("No tokenizer found, language will be first be detected for each audio file (increases inference time).")
            language = language or self.detect_language(audio)
            languages_identified.add(language)
            task = task or "transcribe"
            self.tokenizer = faster_whisper.tokenizer.Tokenizer(self.model.hf_tokenizer,
                                                                self.model.model.is_multilingual, task=task,
                                                                language=language)
        else:
            language = language or self.tokenizer.language_code
            languages_identified.add(language)
            print(f"languages_identified: {languages_identified} and count is: {len(list(languages_identified))}")
            task = task or self.tokenizer.task
            if task != self.tokenizer.task or language != self.tokenizer.language_code:
                print("Tokenizer task or language does not match, reverting to preset tokenizer.")
                print(f"Tokenizer task: {task}, language: {language}")
                self.tokenizer = faster_whisper.tokenizer.Tokenizer(self.model.hf_tokenizer,
                                                                    self.model.model.is_multilingual, task=task,
                                                                    language=language)
                
        print(f"Using tokenizer with task: {task} and language: {language}")
        if self.suppress_numerals:
            previous_suppress_tokens = self.options.suppress_tokens
            numeral_symbol_tokens = find_numeral_symbol_tokens(self.tokenizer)
            print(f"Suppressing numeral and symbol tokens")
            new_suppressed_tokens = numeral_symbol_tokens + self.options.suppress_tokens
            new_suppressed_tokens = list(set(new_suppressed_tokens))
            self.options = self.options._replace(suppress_tokens=new_suppressed_tokens)

        
        segments: List[SingleSegment] = []
        batch_size = batch_size or self._batch_size
        total_segments = len(vad_segments)
        for idx, out in enumerate(self.__call__(data(audio, vad_segments), batch_size=batch_size, num_workers=num_workers,language = language)):
            if print_progress:
                base_progress = ((idx + 1) / total_segments) * 100
                percent_complete = base_progress / 2 if combined_progress else base_progress
                print(f"Progress: {percent_complete:.2f}%...")
            text = out['text']
            if batch_size in [0, 1, None]:
                text = text[0]

            language = self.detect_language(audio[idx*N_SAMPLES:(idx+1)*N_SAMPLES])
            languages_identified.add(language)
            segments.append(
                {
                    "text": text,
                    "start": round(vad_segments[idx]['start'], 3),
                    "end": round(vad_segments[idx]['end'], 3)
                }
            )

            print(f"languages_identified: {languages_identified} and count is: {len(list(languages_identified))}")

        # revert the tokenizer if multilingual inference is enabled
        if self.preset_language is None:
            self.tokenizer = None

        # revert suppressed tokens if suppress_numerals is enabled
        if self.suppress_numerals:
            self.options = self.options._replace(suppress_tokens=previous_suppress_tokens)

        print(f"languages_identified: {languages_identified} and count is: {len(list(languages_identified))}")            

        return {"segments": segments, "language": languages_identified}


    '''
    solution for language detection for every chunk - not effective in transcription
    '''
    # def transcribe(
    #     self, audio: Union[str, np.ndarray], batch_size=None, num_workers=0, task=None, chunk_size=30,
    #     print_progress=False, combined_progress=False
    # ):
    #     if isinstance(audio, str):
    #         audio = load_audio(audio)

    #     def data(audio, segments):
    #         for seg in segments:
    #             f1 = int(seg['start'] * SAMPLE_RATE)
    #             f2 = int(seg['end'] * SAMPLE_RATE)
    #             yield {'inputs': audio[f1:f2]}

    #     vad_segments = self.vad_model({"waveform": torch.from_numpy(audio).unsqueeze(0), "sample_rate": SAMPLE_RATE})
    #     vad_segments = merge_chunks(
    #         vad_segments,
    #         chunk_size,
    #         onset=self._vad_params["vad_onset"],
    #         offset=self._vad_params["vad_offset"],
    #     )
    #     print('vad_segments: ',vad_segments)

    #     segments: List[SingleSegment] = []
    #     for idx, vad_segment in enumerate(vad_segments):
    #         segment_audio = audio[int(vad_segment['start'] * SAMPLE_RATE):int(vad_segment['end'] * SAMPLE_RATE)]

    #         language = self.detect_language(segment_audio)
    #         task = task or "transcribe"

    #         self.tokenizer = faster_whisper.tokenizer.Tokenizer(self.model.hf_tokenizer,
    #                                                        True, task=task,
    #                                                        language=language)
    #         vad_segment = [vad_segment]
    #         print('vad_SEgment',vad_segment)
    #         for idx, out in enumerate(self.__call__(data(segment_audio, vad_segment), batch_size=batch_size, num_workers=num_workers)):
    #             text = out['text']
    #             # if batch_size in [0, 1, None]:
    #             #     text = text[0]
    #             segments.append(
    #                 {
    #                     "text": text,
    #                     "start": round(vad_segment[idx]['start'], 3),
    #                     "end": round(vad_segment[idx]['end'], 3),
    #                     "language":self.detect_language(audio[idx*N_SAMPLES:(idx+1)*N_SAMPLES])
    #                 }
    #             )

    #         if print_progress:
    #             base_progress = ((idx + 1) / len(vad_segments)) * 100
    #             percent_complete = base_progress / 2 if combined_progress else base_progress
    #             print(f"Progress: {percent_complete:.2f}%...")
    #         print(segments)

    #     return {"segments": segments}


    def detect_language(self, audio: np.ndarray):
        print("detect_language in WhisperXPipeline...")
        # if audio.shape[0] < N_SAMPLES:
        #     print("Warning: audio is shorter than 30s, language detection may be inaccurate.")
        model_n_mels = self.model.feat_kwargs.get("feature_size")
        segment = log_mel_spectrogram(audio[:N_SAMPLES],
                                      n_mels=model_n_mels if model_n_mels is not None else 80,
                                      padding=0 if audio.shape[0] >= N_SAMPLES else N_SAMPLES - audio.shape[0])

        encoder_output = self.model.encode(segment)
        results = self.model.model.detect_language(encoder_output)
        print("lang_prob: ",results[0])

        #lang_prob:  [('<|ur|>', 0.6630859375), ('<|hi|>', 0.2255859375), ('<|en|>', 0.023590087890625), ('<|sd|>', 0.0146484375), ('<|mi|>', 0.00821685791015625), ('<|bn|>', 0.00719451904296875), ('<|jw|>', 0.006916046142578125), ('<|pa|>', 0.005645751953125), ('<|ar|>', 0.0034236907958984375), ('<|ne|>', 0.0026874542236328125), ('<|da|>', 0.002506256103515625), ('<|fa|>', 0.002506256103515625), ('<|sa|>', 0.002227783203125), ('<|la|>', 0.0018911361694335938), ('<|de|>', 0.0017910003662109375), ('<|cy|>', 0.00176239013671875), ('<|es|>', 0.001708984375), ('<|ps|>', 0.0016956329345703125), ('<|ms|>', 0.001567840576171875), ('<|nn|>', 0.0013513565063476562), ('<|ta|>', 0.0013408660888671875), ('<|haw|>', 0.0012798309326171875), ('<|sn|>', 0.0011653900146484375), ('<|my|>', 0.0010528564453125), ('<|mr|>', 0.0010280609130859375), ('<|gl|>', 0.0009002685546875), ('<|pt|>', 0.0008525848388671875), ('<|ru|>', 0.0008196830749511719), ('<|uk|>', 0.0008006095886230469), ('<|tr|>', 0.0007944107055664062), ('<|te|>', 0.0006537437438964844), ('<|yo|>', 0.0006093978881835938), ('<|be|>', 0.0005636215209960938), ('<|ja|>', 0.0005335807800292969), ('<|nl|>', 0.0004973411560058594), ('<|it|>', 0.0004782676696777344), ('<|gu|>', 0.0004634857177734375), ('<|br|>', 0.0004353523254394531), ('<|cs|>', 0.0003902912139892578), ('<|zh|>', 0.0003783702850341797), ('<|ko|>', 0.00033783912658691406), ('<|yi|>', 0.00031113624572753906), ('<|bs|>', 0.00030517578125), ('<|si|>', 0.00029921531677246094), ('<|fo|>', 0.00029468536376953125), ('<|fr|>', 0.0002865791320800781), ('<|el|>', 0.0002651214599609375), ('<|bo|>', 0.00022852420806884766), ('<|hy|>', 0.00021982192993164062), ('<|pl|>', 0.0002040863037109375), ('<|eu|>', 0.00019860267639160156), ('<|ro|>', 0.0001678466796875), ('<|af|>', 0.0001558065414428711), ('<|vi|>', 0.0001475811004638672), ('<|ht|>', 0.00013017654418945312), ('<|yue|>', 0.0001246929168701172), ('<|ml|>', 0.00011903047561645508), ('<|oc|>', 0.00011581182479858398), ('<|as|>', 0.00011539459228515625), ('<|km|>', 0.00011092424392700195), ('<|he|>', 0.0001042485237121582), ('<|az|>', 9.60230827331543e-05), ('<|lo|>', 9.566545486450195e-05), ('<|tl|>', 8.344650268554688e-05), ('<|th|>', 8.308887481689453e-05), ('<|kn|>', 7.593631744384766e-05), ('<|sw|>', 6.008148193359375e-05), ('<|id|>', 5.84721565246582e-05), ('<|ln|>', 5.137920379638672e-05), ('<|is|>', 3.88026237487793e-05), ('<|sr|>', 3.600120544433594e-05), ('<|mn|>', 2.962350845336914e-05), ('<|sv|>', 2.872943878173828e-05), ('<|sl|>', 2.7835369110107422e-05), ('<|no|>', 2.771615982055664e-05), ('<|ca|>', 2.6047229766845703e-05), ('<|ka|>', 2.372264862060547e-05), ('<|bg|>', 2.3365020751953125e-05), ('<|kk|>', 1.823902130126953e-05), ('<|mk|>', 1.424551010131836e-05), ('<|sq|>', 1.33514404296875e-05), ('<|sk|>', 8.463859558105469e-06), ('<|so|>', 7.569789886474609e-06), ('<|lv|>', 7.212162017822266e-06), ('<|tg|>', 7.152557373046875e-06), ('<|hu|>', 6.079673767089844e-06), ('<|fi|>', 4.649162292480469e-06), ('<|hr|>', 4.410743713378906e-06), ('<|am|>', 3.2186508178710938e-06), ('<|mt|>', 2.2649765014648438e-06), ('<|su|>', 1.7881393432617188e-06), ('<|et|>', 1.3113021850585938e-06), ('<|uz|>', 9.5367431640625e-07), ('<|ha|>', 8.940696716308594e-07), ('<|tt|>', 8.344650268554688e-07), ('<|mg|>', 5.364418029785156e-07), ('<|lt|>', 4.76837158203125e-07), ('<|tk|>', 4.172325134277344e-07), ('<|ba|>', 3.5762786865234375e-07), ('<|lb|>', 2.980232238769531e-07)]

        selected_language = 'hi' #default to hindi
        selected_language_probability = 0.0
        allowed_languages = ['en', 'hi', 'bn', 'te', 'mr', 'ta', 'gu', 'kn', 'or', 'pa', 'ml']

        for language_token, language_probability in results[0]:
            language = language_token[2:-2]
            if language in allowed_languages:
                selected_language = language
                selected_language_probability = language_probability
                break

        print(f"Detected language: {selected_language} ({selected_language_probability:.2f}) in the 8s chunk of audio...")
        return selected_language

def load_model(whisper_arch,
               device,
               device_index=0,
               compute_type="float16",
               asr_options=None,
               language : Optional[str] = None,
               vad_model=None,
               vad_options=None,
               model : Optional[WhisperModel] = None,
               task="transcribe",
               download_root=None,
               threads=4):
    '''Load a Whisper model for inference.
    Args:
        whisper_arch: str - The name of the Whisper model to load.
        device: str - The device to load the model on.
        compute_type: str - The compute type to use for the model.
        options: dict - A dictionary of options to use for the model.
        language: str - The language of the model. (use English for now)
        model: Optional[WhisperModel] - The WhisperModel instance to use.
        download_root: Optional[str] - The root directory to download the model to.
        threads: int - The number of cpu threads to use per worker, e.g. will be multiplied by num workers.
    Returns:
        A Whisper pipeline.
    '''

    if whisper_arch.endswith(".en"):
        language = "en"

    model = model or WhisperModel(whisper_arch,
                         device=device,
                         device_index=device_index,
                         compute_type=compute_type,
                         download_root=download_root,
                         cpu_threads=threads)
    if language is not None:
        tokenizer = faster_whisper.tokenizer.Tokenizer(model.hf_tokenizer, model.model.is_multilingual, task=task, language=language)
    else:
        print("No language specified, language will be first be detected for each audio file (increases inference time).")
        tokenizer = None

    default_asr_options =  {
        "beam_size": 5,
        "best_of": 5,
        "patience": 1,
        "length_penalty": 1,
        "repetition_penalty": 1,
        "no_repeat_ngram_size": 0,
        "temperatures": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "compression_ratio_threshold": 2.4,
        "log_prob_threshold": -1.0,
        "no_speech_threshold": 0.6,
        "condition_on_previous_text": False,
        "prompt_reset_on_temperature": 0.5,
        "initial_prompt": None,
        "prefix": None,
        "suppress_blank": True,
        "suppress_tokens": [-1],
        "without_timestamps": True,
        "max_initial_timestamp": 0.0,
        "word_timestamps": False,
        "prepend_punctuations": "\"'“¿([{-",
        "append_punctuations": "\"'.。,，!！?？:：”)]}、",
        "suppress_numerals": False,
        "max_new_tokens": None,
        "clip_timestamps": None,
        "hallucination_silence_threshold": None,
    }

    if asr_options is not None:
        default_asr_options.update(asr_options)

    suppress_numerals = default_asr_options["suppress_numerals"]
    del default_asr_options["suppress_numerals"]

    default_asr_options = faster_whisper.transcribe.TranscriptionOptions(**default_asr_options)

    default_vad_options = {
        "vad_onset": 0.500,
        "vad_offset": 0.363
    }

    if vad_options is not None:
        default_vad_options.update(vad_options)

    if vad_model is not None:
        vad_model = vad_model
    else:
        vad_model = load_vad_model(torch.device(device), use_auth_token=None, **default_vad_options)

    return FasterWhisperPipeline(
        model=model,
        vad=vad_model,
        options=default_asr_options,
        tokenizer=tokenizer,
        language=language,
        suppress_numerals=suppress_numerals,
        vad_params=default_vad_options,
    )
