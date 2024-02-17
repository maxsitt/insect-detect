import depthai as dai
from model_config import load_model_config

def create_pipeline(four_k_resolution):
    model_path, config = load_model_config()

    # Get detection model metadata from config JSON
    nn_config = config.get("nn_config", {})
    nn_metadata = nn_config.get("NN_specific_metadata", {})
    classes = nn_metadata.get("classes", {})
    coordinates = nn_metadata.get("coordinates", {})
    anchors = nn_metadata.get("anchors", {})
    anchor_masks = nn_metadata.get("anchor_masks", {})
    iou_threshold = nn_metadata.get("iou_threshold", {})
    confidence_threshold = nn_metadata.get("confidence_threshold", {})
    nn_mappings = config.get("mappings", {})
    labels = nn_mappings.get("labels", {})

    # Create depthai pipeline
    pipeline = dai.Pipeline()

    # Create and configure camera node
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    #cam_rgb.initialControl.setAutoFocusLensRange(142,146) # platform ~9.5 inches from the camera
    #cam_rgb.initialControl.setManualFocus(143) # platform ~9.5 inches from the camera
    cam_rgb.initialControl.setManualExposure(80000,400)
    #cam_rgb.setImageOrientation(dai.CameraImageOrientation.ROTATE_180_DEG)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)


    if not four_k_resolution:
        cam_rgb.setIspScale(1, 2) # downscale 4K to 1080p HQ frames (1920x1080 px)
    cam_rgb.setPreviewSize(320, 320) # downscaled LQ frames for model input
    cam_rgb.setPreviewKeepAspectRatio(False) # "squeeze" frames (16:9) to square (1:1)
    cam_rgb.setInterleaved(False) # planar layout
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

    cam_rgb.setFps(10) # frames per second available for focus/exposure/model input

    # Create detection network node and define input
    nn = pipeline.create(dai.node.YoloDetectionNetwork)
    cam_rgb.preview.link(nn.input) # downscaled LQ frames as model input
    nn.input.setBlocking(False)

    # Set detection model specific settings
    nn.setBlobPath(model_path)
    nn.setNumClasses(classes)
    nn.setCoordinateSize(coordinates)
    nn.setAnchors(anchors)
    nn.setAnchorMasks(anchor_masks)
    nn.setIouThreshold(iou_threshold)
    nn.setConfidenceThreshold(confidence_threshold)
    nn.setNumInferenceThreads(2)

    # Create and configure object tracker node and define inputs
    tracker = pipeline.create(dai.node.ObjectTracker)
    tracker.setTrackerType(dai.TrackerType.ZERO_TERM_IMAGELESS)
    #tracker.setTrackerType(dai.TrackerType.SHORT_TERM_IMAGELESS) # better for low fps
    tracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.UNIQUE_ID)
    nn.passthrough.link(tracker.inputTrackerFrame)
    nn.passthrough.link(tracker.inputDetectionFrame)
    nn.out.link(tracker.inputDetections)

    # Create script node and define inputs
    script = pipeline.create(dai.node.Script)
    script.setProcessor(dai.ProcessorType.LEON_CSS)
    cam_rgb.video.link(script.inputs["frames"]) # HQ frames
    script.inputs["frames"].setBlocking(False)
    tracker.out.link(script.inputs["tracker"]) # tracklets + passthrough detections
    script.inputs["tracker"].setBlocking(False)

    # Set script that will be run on-device (Luxonis OAK)
    script.setScript('''
    # Create empty list to save HQ frames + sequence numbers
    lst = []

    def get_synced_frame(track_seq):
        """Compare tracker with frame sequence number and send frame if equal."""
        global lst
        for i, frame in enumerate(lst):
            if track_seq == frame.getSequenceNum():
                lst = lst[i:]
                break
        return lst[0]

    # Sync tracker output with HQ frames
    while True:
        lst.append(node.io["frames"].get())
        tracks = node.io["tracker"].tryGet()
        if tracks is not None:
            track_seq = node.io["tracker"].get().getSequenceNum()
            if len(lst) == 0: continue
            node.io["frame_out"].send(get_synced_frame(track_seq))
            node.io["track_out"].send(tracks)
            lst.pop(0) # remove synchronized frame from the list
    ''')

    # Define script node outputs
    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("frame")
    script.outputs["frame_out"].link(xout_rgb.input) # synced HQ frames

    xout_tracker = pipeline.create(dai.node.XLinkOut)
    xout_tracker.setStreamName("track")
    script.outputs["track_out"].link(xout_tracker.input) # synced tracker output

    return pipeline, labels