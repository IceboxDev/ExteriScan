import argparse
import asyncio
import json
import logging
import os
import ssl
import uuid

import cv2
from aiohttp import web
from av import VideoFrame

from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder

import numpy as np
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

sys.path.append(r'../models-master/research')
sys.path.append(r'../models-master/research/slim')
sys.path.append(r'../models-master/research/object_detection')

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from utils import label_map_util
from utils import visualization_utils as vis_util

ROOT = os.path.dirname(__file__)

logger = logging.getLogger("pc")
pcs = set()


# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = 'frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('', 'label_map.pbtxt')

# Number of classes to detect
NUM_CLASSES = 90

# Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


# Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# Helper code
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)



# Detection
with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        def evalImage(image_np):
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # Extract image tensor
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Extract detection boxes
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Extract detection scores
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            # Extract detection classes
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            # Extract number of detectionsd
            num_detections = detection_graph.get_tensor_by_name(
                'num_detections:0')
            # Actual detection.
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)
            
            return image_np

        print("initializing")
        evalImage(np.zeros((512,512,3), np.uint8))
        print("init done")

        cv2.namedWindow("ExteriScan", cv2.WINDOW_NORMAL)
        cv2.waitKey(1)

        class VideoTransformTrack(MediaStreamTrack):
            """
            A video stream track that transforms frames from an another track.
            """

            kind = "video"
            receiving = False

            def __init__(self, track, transform):
                super().__init__()  # don't forget this!
                self.track = track
                self.transform = transform

            async def recv(self):
                if self.receiving:
                    return
                self.receiving = True
                frame = await self.track.recv()

                if self.transform == "detection":
                    image_np = frame.to_ndarray(format="bgr24")

                    image_np = evalImage(image_np)

                    # Display output
                    cv2.imshow('ExteriScan', image_np)
                    cv2.waitKey(1)

                    new_frame = VideoFrame.from_ndarray(image_np, format="bgr24")
                    new_frame.pts = frame.pts
                    new_frame.time_base = frame.time_base
                    self.receiving = False
                    return new_frame
                else:
                    self.receiving = False
                    return frame


        async def index(request):
            content = open(os.path.join(ROOT, "index.html"), "r").read()
            return web.Response(content_type="text/html", text=content)


        async def javascript(request):
            content = open(os.path.join(ROOT, "client.js"), "r").read()
            return web.Response(content_type="application/javascript", text=content)


        async def offer(request):
            params = await request.json()
            offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

            pc = RTCPeerConnection()
            pc_id = "PeerConnection(%s)" % uuid.uuid4()
            pcs.add(pc)

            def log_info(msg, *args):
                logger.info(pc_id + " " + msg, *args)

            log_info("Created for %s", request.remote)

            # prepare local media
            player = MediaPlayer(os.path.join(ROOT, "demo-instruct.wav"))
            if args.write_audio:
                recorder = MediaRecorder(args.write_audio)
            else:
                recorder = MediaBlackhole()

            @pc.on("datachannel")
            def on_datachannel(channel):
                @channel.on("message")
                def on_message(message):
                    if isinstance(message, str) and message.startswith("ping"):
                        channel.send("pong" + message[4:])

            @pc.on("iceconnectionstatechange")
            async def on_iceconnectionstatechange():
                log_info("ICE connection state is %s", pc.iceConnectionState)
                if pc.iceConnectionState == "failed":
                    await pc.close()
                    pcs.discard(pc)

            @pc.on("track")
            def on_track(track):
                log_info("Track %s received", track.kind)

                if track.kind == "audio":
                    pc.addTrack(player.audio)
                    recorder.addTrack(track)
                elif track.kind == "video":
                    local_video = VideoTransformTrack(
                        track, transform=params["video_transform"]
                    )
                    pc.addTrack(local_video)

                @track.on("ended")
                async def on_ended():
                    log_info("Track %s ended", track.kind)
                    await recorder.stop()

            # handle offer
            await pc.setRemoteDescription(offer)
            await recorder.start()

            # send answer
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)

            return web.Response(
                content_type="application/json",
                text=json.dumps(
                    {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
                ),
            )


        async def on_shutdown(app):
            # close peer connections
            coros = [pc.close() for pc in pcs]
            await asyncio.gather(*coros)
            pcs.clear()


        if __name__ == "__main__":
            parser = argparse.ArgumentParser(
                description="WebRTC audio / video / data-channels demo"
            )
            parser.add_argument("--cert-file", help="SSL certificate file (for HTTPS)")
            parser.add_argument("--key-file", help="SSL key file (for HTTPS)")
            parser.add_argument(
                "--port", type=int, default=8080, help="Port for HTTP server (default: 8080)"
            )
            parser.add_argument("--verbose", "-v", action="count")
            parser.add_argument("--write-audio", help="Write received audio to a file")
            args = parser.parse_args()

            if args.verbose:
                logging.basicConfig(level=logging.DEBUG)
            else:
                logging.basicConfig(level=logging.INFO)

            if args.cert_file:
                ssl_context = ssl.SSLContext()
                ssl_context.load_cert_chain(args.cert_file, args.key_file)
            else:
                ssl_context = None

            app = web.Application()
            app.on_shutdown.append(on_shutdown)
            app.router.add_get("/", index)
            app.router.add_get("/client.js", javascript)
            app.router.add_post("/offer", offer)
            web.run_app(app, access_log=None, port=args.port, ssl_context=ssl_context)
