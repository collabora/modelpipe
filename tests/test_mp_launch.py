"""Unit tests for `mp-launch`'s gst -> g2g pipeline rewrite.

These run the real `element_shells` scan over the real plugin directory, so a
plugin that stops declaring `register_gst_element` fails a test rather than
silently dropping out of the map.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "plugins" / "python"))

import mp_launch  # noqa: E402


@pytest.fixture(scope="module")
def shells():
    return mp_launch.element_shells()


def test_element_shells_finds_the_hosted_elements(shells):
    assert shells["mp_yolo"] == mp_launch.ElementShell(
        "yolo", "YOLOTransform", reads_device_frames=True
    )
    assert shells["mp_objectdetector"].module == "objectdetector"
    assert shells["mp_overlay"] == mp_launch.ElementShell("overlay", "Overlay")
    assert shells["mp_anonymize"] == mp_launch.ElementShell(
        "anonymize", "Anonymize", reads_device_frames=True
    )


def test_an_element_several_chains_feed_becomes_the_batching_host(shells):
    pipeline = (
        "videotestsrc ! mp_yolo name=detector "
        "videotestsrc ! detector. videotestsrc ! detector. "
        "detector. ! fakesink"
    ).split()
    rewritten = mp_launch.rewrite_for_g2g(pipeline, shells)
    assert "pyaggregator" in rewritten
    assert "pyelement" not in rewritten


def test_a_single_input_element_stays_on_the_one_in_host(shells):
    # Whisper derives from GstBase.Aggregator but takes one chain, which g2g
    # hosts as a transform.
    assert (
        mp_launch.rewrite_segment("mp_whispertranscribe language=ko", shells)[0]
        == "pyelement"
    )


def test_element_shells_resolves_a_name_given_as_a_class_constant(shells):
    # kafkasink registers under `KafkaSink.GST_PLUGIN_NAME`, not a literal.
    assert shells["mp_kafkasink"] == mp_launch.ElementShell(
        "kafkasink", "KafkaSink", reads_device_frames=True, is_sink=True
    )


def test_hosted_element_becomes_pyelement_keeping_its_properties(shells):
    assert mp_launch.rewrite_segment(
        "mp_yolo model-name=yolo11m device=cuda:0", shells
    ) == [
        "pyelement",
        "module=yolo",
        "class=YOLOTransform",
        "model-name=yolo11m",
        "device=cuda:0",
    ]


def test_overlay_counter_is_hosted_keeping_its_properties(shells):
    assert mp_launch.rewrite_segment(
        "mp_overlay_counter meta-path=data/sample_metadata.json tracking=true",
        shells,
    ) == [
        "pyelement",
        "module=overlay_counter",
        "class=OverlayCounter",
        "meta-path=data/sample_metadata.json",
        "tracking=true",
    ]


def test_overlay_becomes_the_native_element(shells):
    assert mp_launch.rewrite_segment("mp_overlay", shells) == ["analyticsoverlay"]


def test_overlay_properties_carry_over_under_the_native_name(shells):
    assert mp_launch.rewrite_segment("mp_overlay tracking=True", shells) == [
        "analyticsoverlay",
        "show-track=True",
    ]


def test_metasink_becomes_the_native_element(shells):
    assert mp_launch.rewrite_segment("mp_metasink location=people.jsonl", shells) == [
        "metasink",
        "location=people.jsonl",
    ]


def test_metareplay_becomes_the_native_element(shells):
    assert mp_launch.rewrite_segment("mp_metareplay location=people.jsonl", shells) == [
        "metareplay",
        "location=people.jsonl",
    ]


def test_alert_becomes_the_native_element(shells):
    assert mp_launch.rewrite_segment(
        ["mp_alert", "cooldown=5", "draw-alert=false", "webhook-url=http://host/a"],
        shells,
    ) == [
        "analyticsalert",
        "cooldown=5",
        "draw-alert=false",
        "webhook-url=http://host/a",
    ]


def test_an_alert_property_with_no_g2g_counterpart_is_refused(shells):
    # g2g runs no MQTT, so the broker cannot be carried over or dropped.
    with pytest.raises(SystemExit, match="mqtt-broker"):
        mp_launch.rewrite_segment("mp_alert mqtt-broker=host", shells)


def test_alertrecorder_becomes_the_native_element(shells):
    assert mp_launch.rewrite_segment(
        [
            "mp_alertrecorder",
            "location=alert-%s.avi",
            "encoder=vp8enc ! webmmux",
            "seconds-before=2",
            "seconds-after=3",
        ],
        shells,
    ) == [
        "alertrecorder",
        "location=alert-%s.avi",
        r'encoder="vp8enc \! webmmux"',
        "seconds-before=2",
        "seconds-after=3",
    ]


def test_digest_becomes_the_native_element(shells):
    assert mp_launch.rewrite_segment("mp_digest window-seconds=5", shells) == [
        "textdigest",
        "window-seconds=5",
    ]


def test_embeddingsink_becomes_the_native_element(shells):
    assert mp_launch.rewrite_segment(
        "mp_embeddingsink location=index.db source-id=camera model-name=clip", shells
    ) == [
        "embeddingsink",
        "location=index.db",
        "source-id=camera",
        "model-name=clip",
    ]


def test_an_overlay_property_with_no_counterpart_is_refused(shells):
    with pytest.raises(SystemExit, match="meta-path"):
        mp_launch.rewrite_segment(
            "mp_overlay meta-path=data/sample_metadata.json", shells
        )


def test_unknown_mp_element_is_refused(shells):
    with pytest.raises(SystemExit, match="mp_nonesuch"):
        mp_launch.rewrite_segment("mp_nonesuch", shells)


def test_shared_elements_pass_through_untouched(shells):
    for segment in ("filesrc location=data/people.mp4", "decodebin", "videoscale"):
        assert mp_launch.rewrite_segment(segment, shells) == segment.split()


def test_raw_video_caps_gain_a_format_only_when_they_lack_one(shells):
    assert mp_launch.rewrite_segment("video/x-raw,width=640,height=480", shells) == [
        "video/x-raw,width=640,height=480,format=RGBA"
    ]
    pinned = "video/x-raw,format=NV12,width=640,height=480"
    assert mp_launch.rewrite_segment(pinned, shells) == [pinned]


def test_caps_written_with_spaces_become_one_token(shells):
    assert mp_launch.rewrite_segment(
        ["video/x-raw,", "width=320,", "height=240"], shells
    ) == ["video/x-raw,width=320,height=240,format=RGBA"]


def test_a_property_value_with_spaces_keeps_them(shells):
    assert mp_launch.rewrite_segment(
        ["mp_clip", "labels=person, bicycle, car", "top-k=3"], shells
    ) == [
        "pyelement",
        "module=clip",
        "class=CLIPTransform",
        'labels="person, bicycle, car"',
        "top-k=3",
    ]


def test_spaces_round_a_property_equals_sign_are_closed_up(shells):
    rewritten = mp_launch.rewrite_for_g2g(
        ["mp_whispertranscribe", "initial_prompt", "=", "Air Traffic Control"],
        shells,
    )
    assert rewritten[-1] == 'initial-prompt="Air Traffic Control"'


def test_a_property_name_is_written_with_dashes(shells):
    assert (
        mp_launch.rewrite_segment(
            ["mp_whispertranscribe", "initial_prompt=Air_Traffic"], shells
        )[-1]
        == "initial-prompt=Air_Traffic"
    )


def test_a_value_after_a_spaced_equals_sign_is_not_read_as_a_reference(shells):
    assert mp_launch.rewrite_for_g2g(
        ["mp_yolo", "model-name", "=", "yolo11m.pt", "!", "fakesink"], shells
    )[-3:] == ["model-name=yolo11m.pt", "!", "fakesink"]


def test_a_property_value_keeps_the_quotes_inside_it(shells):
    assert (
        mp_launch.rewrite_segment(["mp_alert", 'rules={"class":"person"}'], shells)[-1]
        == r"rules={\"class\":\"person\"}"
    )


def test_a_sink_drops_the_clock_properties_g2g_has_no_counterpart_for(shells):
    assert mp_launch.rewrite_segment("fakesink async=0 sync=0", shells) == ["fakesink"]


def test_a_sink_asked_to_wait_on_the_clock_is_refused(shells):
    with pytest.raises(SystemExit, match="clocksync"):
        mp_launch.rewrite_segment("autovideosink sync=true", shells)


def test_pipewiresrc_drops_the_timestamping_g2g_always_does(shells):
    assert mp_launch.rewrite_segment("pipewiresrc do-timestamp=true", shells) == [
        "pipewiresrc"
    ]


def test_pipewiresrc_asked_not_to_timestamp_is_refused(shells):
    with pytest.raises(SystemExit, match="capture time"):
        mp_launch.rewrite_segment("pipewiresrc do-timestamp=false", shells)


def test_a_hosted_element_named_like_a_sink_keeps_its_own_properties(shells):
    # `sync` on a g2g sink names a clock wait that does not exist there, but on
    # `mp_kafkasink` it is the Kafka producer's own knob.
    assert mp_launch.rewrite_segment("mp_kafkasink sync=true", shells) == [
        "pysink",
        "module=kafkasink",
        "class=KafkaSink",
        "sync=true",
    ]


def test_a_hosted_sink_becomes_pysink_keeping_its_properties(shells):
    pipeline = "mp_kafkasink broker=localhost:29092 topic=t".split()
    assert mp_launch.rewrite_for_g2g(pipeline, shells) == [
        "pysink",
        "module=kafkasink",
        "class=KafkaSink",
        "broker=localhost:29092",
        "topic=t",
    ]


def test_a_hosted_sink_after_one_on_cuda_frames_states_nv12_in_its_caps(shells):
    # pysink has no `format` property, it takes the format in its input caps.
    pipeline = "nvdec ! mp_yolo device=cuda ! mp_kafkasink topic=t".split()
    assert mp_launch.rewrite_for_g2g(pipeline, shells)[-6:] == [
        "pysink",
        "module=kafkasink",
        "class=KafkaSink",
        "cuda-frames=true",
        "input-caps=video/x-raw,format=NV12",
        "topic=t",
    ]


def test_textoverlay_drops_the_wait_it_never_does(shells):
    assert mp_launch.rewrite_segment(
        "textoverlay name=overlay wait-text=false", shells
    ) == ["textoverlay", "name=overlay"]


def test_whole_documented_pipeline_rewrites(shells):
    pipeline = (
        "filesrc location=data/soccer_tracking.mp4 ! decodebin ! videoconvert "
        "! videoscale ! video/x-raw,width=640,height=480 "
        "! mp_yolo model-name=yolo11m device=cuda:0 track=True "
        "! mp_overlay ! videoconvert ! autovideosink"
    )
    assert (
        mp_launch.rewrite_for_g2g(pipeline.split(), shells)
        == (
            "filesrc location=data/soccer_tracking.mp4 ! decodebin ! videoconvert "
            "! videoscale ! video/x-raw,width=640,height=480,format=RGBA "
            "! pyelement module=yolo class=YOLOTransform model-name=yolo11m "
            "device=cuda:0 track=True "
            "! analyticsoverlay ! videoconvert ! autovideosink"
        ).split()
    )


def test_the_two_stream_detector_rewrites_onto_the_native_mux_and_demux(shells):
    source = (
        "filesrc location=data/soccer_tracking.mp4 ! decodebin ! videoconvertscale "
        "! video/x-raw,width=640,height=480,format=RGB"
    )
    branch = "queue ! videoconvert ! {} ! videoconvert ! autovideosink"
    pipeline = (
        f"{source} ! mp_streammux name=mux {source} ! mux. "
        "mux. ! mp_yolo model-name=yolo11m device=cuda:0 track=True "
        "! mp_streamdemux name=demux "
        f"demux. ! {branch.format('mp_overlay')} sync=false "
        f"demux. ! {branch.format('mp_overlay')} sync=false"
    )
    assert (
        mp_launch.rewrite_for_g2g(pipeline.split(), shells)
        == (
            f"{source} ! streammux name=mux {source} ! mux. "
            "mux. ! pyelement module=yolo class=YOLOTransform format=RGB "
            "model-name=yolo11m device=cuda:0 track=True "
            "! streamdemux name=demux "
            f"demux. ! {branch.format('analyticsoverlay')} "
            f"demux. ! {branch.format('analyticsoverlay')}"
        ).split()
    )


def test_the_mux_and_demux_keep_the_pad_names_they_are_linked_by(shells):
    source = "videotestsrc ! video/x-raw,width=320,height=240 ! queue"
    pipeline = (
        f"{source} ! mp_streammux name=mux "
        f"{source} ! mux.sink_1 {source} ! mux.sink_2 "
        "mux.src ! queue ! mp_streamdemux name=demux "
        "demux.src_0 ! queue ! fakesink demux.src_1 ! queue ! fakesink "
        "demux.src_2 ! queue ! fakesink"
    )
    pinned = "videotestsrc ! video/x-raw,width=320,height=240,format=RGBA ! queue"
    assert (
        mp_launch.rewrite_for_g2g(pipeline.split(), shells)
        == (
            f"{pinned} ! streammux name=mux "
            f"{pinned} ! mux.sink_1 {pinned} ! mux.sink_2 "
            "mux.src ! queue ! streamdemux name=demux "
            "demux.src_0 ! queue ! fakesink demux.src_1 ! queue ! fakesink "
            "demux.src_2 ! queue ! fakesink"
        ).split()
    )


def test_hosted_element_takes_the_format_the_caps_ahead_of_it_pin(shells):
    pipeline = (
        "videoconvert ! video/x-raw,format=RGB,width=640,height=640 "
        "! mp_inference engine-name=onnx"
    ).split()
    assert mp_launch.rewrite_for_g2g(pipeline, shells)[-4:] == [
        "module=inference",
        "class=GenericInferenceTransform",
        "format=RGB",
        "engine-name=onnx",
    ]


def test_the_elements_that_accept_device_frames_are_found(shells):
    reading = {name for name, shell in shells.items() if shell.reads_device_frames}
    assert reading == {
        "mp_classifier",
        "mp_yolo",
        "mp_yolo_pose",
        "mp_depth",
        "mp_superres",
        "mp_tracker",
        "mp_anonymize",
        "mp_football_overlay",
        "mp_baseball_overlay",
        "mp_overlay_counter",
        "mp_kafkasink",
    }


def test_an_element_reading_device_frames_takes_them_straight_from_nvdec(shells):
    pipeline = (
        "filesrc location=data/people.mp4 ! qtdemux ! h264parse ! nvdec "
        "! mp_yolo model-name=yolo11n device=cuda "
        "! videoconvert ! mp_tracker ! mp_overlay ! fakesink"
    )
    assert (
        mp_launch.rewrite_for_g2g(pipeline.split(), shells)
        == (
            "filesrc location=data/people.mp4 ! qtdemux ! h264parse ! nvdec "
            "! pyelement module=yolo class=YOLOTransform cuda-frames=true "
            "format=NV12 model-name=yolo11n device=cuda "
            "! videoconvert ! pyelement module=tracker class=TrackerTransform "
            "! analyticsoverlay ! fakesink"
        ).split()
    )


def test_gst_cuda_converters_after_nvdec_are_dropped(shells):
    pipeline = [
        *"qtdemux ! h264parse ! nvh264dec ! cudaconvert ! cudascale ! capsfilter".split(),
        "caps=video/x-raw(memory:CUDAMemory),format=RGBA,width=640,height=480",
        *"! mp_classifier device=cuda".split(),
    ]
    assert (
        mp_launch.rewrite_for_g2g(pipeline, shells)
        == (
            "qtdemux ! h264parse ! nvh264dec ! pyelement module=classifier "
            "class=Classifier cuda-frames=true format=NV12 device=cuda"
        ).split()
    )


def test_an_element_after_one_on_cuda_frames_takes_them_too(shells):
    pipeline = "nvdec ! mp_yolo device=cuda ! mp_tracker ! fakesink".split()
    assert (
        mp_launch.rewrite_for_g2g(pipeline, shells)
        == (
            "nvdec ! pyelement module=yolo class=YOLOTransform cuda-frames=true "
            "format=NV12 device=cuda ! pyelement module=tracker "
            "class=TrackerTransform cuda-frames=true format=NV12 ! fakesink"
        ).split()
    )


def test_a_converter_after_nvdec_keeps_the_element_on_system_memory(shells):
    pipeline = "nvdec ! videoconvert ! mp_yolo device=cuda".split()
    assert "cuda-frames=true" not in mp_launch.rewrite_for_g2g(pipeline, shells)


def test_an_element_reading_host_frames_takes_no_cuda_frames_after_nvdec(shells):
    pipeline = "nvdec ! mp_inference engine-name=onnx".split()
    assert "cuda-frames=true" not in mp_launch.rewrite_for_g2g(pipeline, shells)


def test_the_batching_host_takes_the_caps_of_an_audio_in_text_out_element(shells):
    assert mp_launch.rewrite_segment(
        "mp_whispertranscribe", shells, host=mp_launch.PY_AGGREGATOR
    ) == [
        "pyaggregator",
        "module=whispertranscribe",
        "class=WhisperTranscribe",
        "input-caps=audio/x-raw,format=S16LE,layout=interleaved,rate=16000,channels=1",
        "output-caps=text/x-raw,format=utf8",
    ]


def test_the_batching_host_takes_the_caps_of_a_text_in_audio_out_element(shells):
    assert mp_launch.rewrite_segment(
        "mp_whisperspeechtts", shells, host=mp_launch.PY_AGGREGATOR
    ) == [
        "pyaggregator",
        "module=whisperspeechtts",
        "class=WhisperSpeechTTS",
        "input-caps=text/x-raw,format=utf8",
        "output-caps=audio/x-raw,format=S16LE,layout=interleaved,rate=24000,channels=1",
    ]


def test_the_one_in_host_takes_the_caps_of_an_audio_in_text_out_element(shells):
    pipeline = (
        "filesrc location=data/audio_sample.wav ! decodebin ! audioconvert "
        "! mp_whispertranscribe language=ko ! fakesink"
    ).split()
    assert (
        mp_launch.rewrite_for_g2g(pipeline, shells)
        == (
            "filesrc location=data/audio_sample.wav ! decodebin ! audioconvert "
            "! pyelement module=whispertranscribe class=WhisperTranscribe "
            "input-caps=audio/x-raw,format=S16LE,layout=interleaved,rate=16000,channels=1 "
            "output-caps=text/x-raw,format=utf8 language=ko "
            "! fakesink"
        ).split()
    )


def test_the_one_in_host_takes_the_caps_of_a_text_in_audio_out_element(shells):
    pipeline = (
        "filesrc location=data/lines.txt ! mp_whisperspeechtts ! fakesink".split()
    )
    assert (
        mp_launch.rewrite_for_g2g(pipeline, shells)
        == (
            "filesrc location=data/lines.txt "
            "! pyelement module=whisperspeechtts class=WhisperSpeechTTS "
            "input-caps=text/x-raw,format=utf8 "
            "output-caps=audio/x-raw,format=S16LE,layout=interleaved,rate=24000,channels=1 "
            "! fakesink"
        ).split()
    )


def test_the_one_in_host_takes_the_text_caps_a_remote_llm_inherits(shells):
    assert mp_launch.rewrite_segment(
        "mp_llm_remote url=http://localhost:11434/api/generate model-name=llama3",
        shells,
    ) == [
        "pyelement",
        "module=llm_remote",
        "class=LlmRemote",
        "input-caps=text/x-raw,format=utf8",
        "output-caps=text/x-raw,format=utf8",
        "url=http://localhost:11434/api/generate",
        "model-name=llama3",
    ]


def test_the_one_in_host_takes_the_text_caps_of_coalescehistory(shells):
    assert mp_launch.rewrite_segment("coalescehistory history-length=10", shells) == [
        "pyelement",
        "module=coalescehistory",
        "class=CoalesceHistory",
        "input-caps=text/x-raw,format=utf8",
        "output-caps=text/x-raw,format=utf8",
        "history-length=10",
    ]


def test_the_one_in_host_takes_the_caps_of_a_text_in_video_out_element(shells):
    assert mp_launch.rewrite_segment("mp_stablediffusion device=cuda", shells) == [
        "pyelement",
        "module=stablediffusion",
        "class=StableDiffusion",
        "input-caps=text/x-raw,format=utf8",
        "output-caps=video/x-raw,width=512,height=512,format=RGBA,framerate=0/1",
        "device=cuda",
    ]


def test_a_leaf_that_restates_a_rate_wins_over_the_base_it_inherits_from(shells):
    # Demucs runs at the base class's 44100, Sepformer restates both pads at 8000.
    assert shells["mp_demucs"].caps == (
        (
            "input-caps",
            "audio/x-raw,format=S16LE,layout=interleaved,rate=44100,channels=1",
        ),
        (
            "output-caps",
            "audio/x-raw,format=S16LE,layout=interleaved,rate=44100,channels=1",
        ),
    )
    assert shells["mp_sepformer"].caps == (
        (
            "input-caps",
            "audio/x-raw,format=S16LE,layout=interleaved,rate=8000,channels=1",
        ),
        (
            "output-caps",
            "audio/x-raw,format=S16LE,layout=interleaved,rate=8000,channels=1",
        ),
    )


def test_a_leaf_inherits_the_pad_it_does_not_restate(shells):
    # WhisperLive transcribes into speech, so only its src pad differs.
    assert shells["mp_whisperlive"].caps == (
        (
            "input-caps",
            "audio/x-raw,format=S16LE,layout=interleaved,rate=16000,channels=1",
        ),
        (
            "output-caps",
            "audio/x-raw,format=S16LE,layout=interleaved,rate=24000,channels=1",
        ),
    )


def test_an_element_declaring_no_caps_gets_no_caps_properties(shells):
    pipeline = (
        "videotestsrc ! mp_yolo name=detector "
        "videotestsrc ! detector. videotestsrc ! detector. "
        "detector. ! fakesink"
    ).split()
    rewritten = mp_launch.rewrite_for_g2g(pipeline, shells)
    assert "pyaggregator" in rewritten
    assert not [token for token in rewritten if token.startswith(("input-", "output-"))]


def test_the_caps_survive_the_rewrite_of_a_whole_pipeline(shells):
    pipeline = (
        "filesrc location=data/audio_sample.wav ! decodebin ! audioconvert "
        "! mp_whispertranscribe name=stt language=ko "
        "audiotestsrc ! stt. audiotestsrc ! stt. stt. ! fakesink"
    ).split()
    assert (
        mp_launch.rewrite_for_g2g(pipeline, shells)
        == (
            "filesrc location=data/audio_sample.wav ! decodebin ! audioconvert "
            "! pyaggregator module=whispertranscribe class=WhisperTranscribe "
            "input-caps=audio/x-raw,format=S16LE,layout=interleaved,rate=16000,channels=1 "
            "output-caps=text/x-raw,format=utf8 name=stt language=ko "
            "audiotestsrc ! stt. audiotestsrc ! stt. stt. ! fakesink"
        ).split()
    )


def g2g_pipeline(*segments):
    """Run a pipeline on the g2g launcher, returning what it printed.

    Skips where there is no build to check against, so the drift check runs for
    whoever has both repos and never blocks whoever has one.
    """
    binary = mp_launch.g2g_binary()
    if not binary:
        pytest.skip("needs a g2g-launch-py build to check the names against")
    return subprocess.run(
        [binary, *segments],
        capture_output=True,
        text=True,
        timeout=120,
    )


#: Enough of a pipeline to negotiate raw video into the element under test.
G2G_VIDEO_SOURCE = (
    "videotestsrc",
    "num-buffers=1",
    "!",
    "videoconvert",
    "!",
    f"video/x-raw,format={mp_launch.G2G_RAW_VIDEO_FORMAT}",
    "!",
)


#: The text chain `textdigest` reads, and the one cue the file holds.
G2G_TEXT_SOURCE = ("subtitlesrc", "location={directory}/cues.srt", "!", "subparse", "!")
SUBTITLE_CUE = "1\n00:00:00,000 --> 00:00:01,000\nhello\n\n"

#: One record in the shape `mp_metasink` writes and `metareplay` reads back.
METADATA_RECORD = {
    "pts": 0.0,
    "detections": [{"label": "person", "x": 1, "y": 2, "w": 3, "h": 4, "score": 0.9}],
}

#: What it takes to run each native element for real: one valid value per
#: property that carries over, the chain ahead of it, and what follows it, which
#: is nothing for a sink. `{directory}` is a temporary directory of the run's own.
NATIVE_PROBES = {
    "mp_overlay": (("tracking=true",), G2G_VIDEO_SOURCE, ("!", "fakesink")),
    "mp_metasink": (
        ("location={directory}/meta.jsonl",),
        G2G_VIDEO_SOURCE,
        (),
    ),
    "mp_metareplay": (
        ("location={directory}/records.jsonl",),
        G2G_VIDEO_SOURCE,
        ("!", "fakesink"),
    ),
    "mp_alert": (
        (
            'rules={"class":"person"}',
            "cooldown=1",
            "draw-alert=true",
            "webhook-url=http://127.0.0.1:1/alert",
        ),
        G2G_VIDEO_SOURCE,
        ("!", "fakesink"),
    ),
    "mp_alertrecorder": (
        (
            "location={directory}/alert-%s.avi",
            "encoder=mjpegenc ! avimux",
            "seconds-before=1",
            "seconds-after=1",
        ),
        G2G_VIDEO_SOURCE,
        ("!", "fakesink"),
    ),
    "mp_digest": (("window-seconds=5",), G2G_TEXT_SOURCE, ("!", "fakesink")),
    "mp_streammux": ((), G2G_VIDEO_SOURCE, ("!", "fakesink")),
    "mp_streamdemux": (
        ("name=demux",),
        (
            *G2G_VIDEO_SOURCE,
            "streammux",
            "name=mux",
            *G2G_VIDEO_SOURCE,
            "mux.",
            "mux.",
            "!",
        ),
        ("demux.", "!", "fakesink", "demux.", "!", "fakesink"),
    ),
    "mp_embeddingsink": (
        (
            "location={directory}/index.db",
            "source-id=camera",
            "model-name=clip-vit-b32",
        ),
        G2G_VIDEO_SOURCE,
        (),
    ),
}


def filled(tokens, directory):
    """The tokens with the run's temporary directory in place of the placeholder.

    `str.replace` rather than `str.format`: a rules table is full of braces.
    """
    return [token.replace("{directory}", str(directory)) for token in tokens]


def test_the_native_element_and_the_names_it_renames_to_still_exist(shells, tmp_path):
    """`NATIVE_EQUIVALENTS` and `NATIVE_PROPERTIES` copy names out of glass2glass,
    and `G2G_RAW_VIDEO_FORMAT` states what those elements negotiate. Nothing here
    notices when any of the three changes there, so run each one and see."""
    (tmp_path / "records.jsonl").write_text(json.dumps(METADATA_RECORD) + "\n")
    (tmp_path / "cues.srt").write_text(SUBTITLE_CUE)

    for element, native in mp_launch.NATIVE_EQUIVALENTS.items():
        properties, source, tail = NATIVE_PROBES[element]
        assert {property.partition("=")[0] for property in properties} - {
            mp_launch.NAME_PROPERTY
        } == set(
            mp_launch.NATIVE_PROPERTIES.get(element, {})
        ), f"{element} carries a property no sample value here covers"
        rewritten = mp_launch.rewrite_segment(
            [element, *filled(properties, tmp_path)], shells
        )
        assert rewritten[0] == native
        result = g2g_pipeline(*filled(source, tmp_path), *rewritten, *tail)
        assert result.returncode == 0, (
            f"{element} rewrites to `{' '.join(rewritten)}`, which g2g "
            f"no longer runs:\n{result.stdout}{result.stderr}"
        )


def test_an_element_read_from_twice_is_not_a_muxer(shells):
    # Two chains start at `cap.`, reading two source pads, which is a fan-out.
    pipeline = (
        "videotestsrc ! mp_caption_qwen name=cap "
        "cap.src ! fakesink cap.text_src ! fakesink"
    ).split()
    rewritten = mp_launch.rewrite_for_g2g(pipeline, shells)
    assert "pyelement" in rewritten
    assert "pyaggregator" not in rewritten
