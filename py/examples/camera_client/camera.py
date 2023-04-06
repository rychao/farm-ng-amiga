# Copyright (c) farm-ng, inc.
#
# Licensed under the Amiga Development Kit License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/farm-ng/amiga-dev-kit/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import argparse
import asyncio

import cv2
import numpy as np
import depthai as dai
import matplotlib from pyplot as plt
from farm_ng.oak import oak_pb2
from farm_ng.oak.camera_client import OakCameraClient
from farm_ng.service import service_pb2
from farm_ng.service.service_client import ClientConfig


async def main(address: str, port: int, stream_every_n: int) -> None:
    # configure the camera client
    config = ClientConfig(address=address, port=port)
    client = OakCameraClient(config)

    # Get the streaming object from the service
    response_stream = client.stream_frames(every_n=stream_every_n)

    while True:
        # check the service state
        state = await client.get_state()
        if state.value != service_pb2.ServiceState.RUNNING:
            print("Camera is not streaming!")
            continue

        response: oak_pb2.StreamFramesReply = await response_stream.read()

        if response:
            # get the sync frame
            frame: oak_pb2.OakSyncFrame = response.frame
            print(f"Got frame: {frame.sequence_num}")
            print(f"Device info: {frame.device_info}")
            print(f"Timestamp: {frame.rgb.meta.timestamp}")
            print("#################################\n")

            try:
                # cast image data bytes to numpy and decode
                # NOTE: explore frame.[rgb, disparity, left, right]
                # image = np.frombuffer(frame.rgb.image_data, dtype="uint8")

                # RGB Image
                # imageRGB = np.frombuffer(frame.RGB.image_data, dtype="uint8")
                # imageRGB = cv2.imdecode(imageRGB, cv2.IMREAD_UNCHANGED)

                # disparity map
                imageDisp = np.frombuffer(frame.disparity.image_data, dtype = "uint8")
                imageDisp = cv2.imdecode(imageDisp, cv2.IMREAD_UNCHANGED)

                # Right image
                imageR = np.frombuffer(frame.left.image_data, dtype="uint8")
                imageR = cv2.imdecode(imageR, cv2.IMREAD_UNCHANGED)

                # Left image
                imageL = np.frombuffer(frame.right.image_data, dtype="uint8")
                imageL = cv2.imdecode(imageL, cv2.IMREAD_UNCHANGED)
                
                # visualize the image
                # cv2.namedWindow("imageDisp", cv2.WINDOW_NORMAL)
                # cv2.imshow("imageDisp", imageDisp)

                # cv2.namedWindow("imageR", cv2.WINDOW_NORMAL)
                # cv2.imshow("imageR", imageR)
                
                # cv2.namedWindow("imageL", cv2.WINDOW_NORMAL)
                # cv2.imshow("imageL", imageL)

                # imshow cannot display RGB simaultaneously with other 3 windows?
                # cv2.namedWindow("imageRGB", cv2.WINDOW_NORMAL)
                # cv2.imshow("imageRGB", imageRGB)

                # DEPTHAI
                # pipeline = dai.Pipeline()
                # stereo = pipeline.create(dai.node.StereoDepth)

                # Initiate and StereoBM object
                stereo = cv2.StereoBM_create(numDisparities=128, blockSize=15)

                # compute the disparity map
                imageLnew = cv2.cvtColor(imageL, cv2.COLOR_BGR2GRAY)
                imageRnew = cv2.cvtColor(imageR, cv2.COLOR_BGR2GRAY)


                disparity = stereo.compute(imageLnew,imageRnew)
                plt.imshow(disparity,'gray')
                plt.show()


                cv2.waitKey(1)
            except Exception as e:
                print(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="amiga-camera-app")
    parser.add_argument("--port", type=int, required=True, help="The camera port.")
    parser.add_argument("--address", type=str, default="localhost", help="The camera address")
    parser.add_argument("--stream-every-n", type=int, default=1, help="Streaming frequency")
    args = parser.parse_args()

    asyncio.run(main(args.address, args.port, args.stream_every_n))
