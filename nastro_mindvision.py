import ctypes
import cv2
import numpy as np
import threading
import mvsdk
import time
import gc


def get_first_available_camera(): 
    devList = mvsdk.CameraEnumerateDevice()
    nDev = len(devList)
    if nDev < 1:
        print("No camera was found!")
        return None

    devInfo = devList[0]
    print(devInfo)
    return devInfo


Tlock = threading.RLock()
camera_instance = None

class MindVisionCamera():
    def __init__(self):
        global camera_instance

        camera_instance = self
        self.continuous_acquire_run_flag = False
        self.img_list = []
        self.frameBuffer = None
        self.hcam = 0
        self.grabber = ctypes.c_voidp(0)
        self.dev = get_first_available_camera()#tSdkCameraDevInfo()
        self.new_frame = False
        if self.dev:
            try:
                self.hcam = mvsdk.CameraInit(self.dev, -1, -1)
                self.cap = mvsdk.CameraGetCapability(self.hcam)
                self.monoCamera = (self.cap.sIspCapacity.bMonoSensor != 0)

                frameBufferSize = self.cap.sResolutionRange.iWidthMax * self.cap.sResolutionRange.iHeightMax * (1 if self.monoCamera else 3)
                self.frameBuffer = mvsdk.CameraAlignMalloc(frameBufferSize, 16)
                mvsdk.CameraPlay(self.hcam)
                #mvsdk.CameraSetTriggerMode(self.hcam, 1)
                #mvsdk.CameraSetCallbackFunction(self.hcam, GrabImageCallback, 0)

            except mvsdk.CameraException as e:
                print("CameraInit Failed({}): {}".format(e.error_code, e.message) )
                return
            
        else:
            print("mindvision_sdk.py - MindVisionCamera.__init__ - Error. No camera found!")

    def set_exposure(self, exposure_microseconds):
        mvsdk.CameraSetAeState(self.hcam, 0)
        mvsdk.CameraSetExposureTime(self.hcam, exposure_microseconds)

    def start(self):
        t = threading.Thread(target = self.continuous_acquire)
        t.setDaemon(True)
        t.start()
        #threading.Timer(1, self.continuous_acquire).start()
        #frameBufferSize = self.cap.sResolutionRange.iWidthMax * self.cap.sResolutionRange.iHeightMax * (1 if self.monoCamera else 3)
        #self.frameBuffer = mvsdk.CameraAlignMalloc(frameBufferSize, 16)
        #mvsdk.CameraPlay(self.hcam)

    def continuous_acquire(self):
        global Tlock
        self.continuous_acquire_run_flag = True
        #mvsdk.CameraPlay(self.hcam)
        while self.continuous_acquire_run_flag:
            try:
                self.trigger_software()
                
                #print("new continuous image")

            except mvsdk.CameraException as e:
                print(e.message)
                if e.error_code != mvsdk.CAMERA_STATUS_TIME_OUT:
                    print("CameraGetImageBuffer failed({}): {}".format(e.error_code, e.message) )
        
    def stop(self):
        self.continuous_acquire_run_flag = False
        
    def get_image_buf(self):
        global Tlock
        if len(self.img_list)<1:
            return None
        #print(len(self.img_list))
        gc.collect()
        with Tlock:
            #img = self.img_list[-1]
            #self.img_list = [img,]
            #img = img.reshape(512, 1280, 3 )
            #ret, jpeg = cv2.imencode('.jpg', img)
            #return jpeg
            buf = self.img_list[:]
            self.img_list = []
            return buf

    def set_io_state(self, n, state):
        mvsdk.CameraSetIOState(self.hcam, n, state)

    def set_trigger_mode(self, mode): #0- continuous, 1-software, 2-hardware 
        mvsdk.CameraSetTriggerMode(self.hcam, mode)

    def trigger_software(self):
        """
        mvsdk.CameraSoftTrigger(self.hcam)
        self.new_frame = False
        t = time.time()
        while not self.new_frame and ((time.time()-t)<1):
            time.sleep(0.001)
        #mvsdk.CameraClearBuffer(self.hcam)
        """
        t = time.time()
        if mvsdk.CameraSoftTrigger(self.hcam) == 0:
            CAMERA_GET_IMAGE_PRIORITY_OLDEST = 0, # Get the oldest frame in the cache
            CAMERA_GET_IMAGE_PRIORITY_NEWEST = 1 #get the latest frame in the cache (all the old frames will be discarded)
            CAMERA_GET_IMAGE_PRIORITY_NEXT = 2 #Discard all frames in the cache, and if the camera is currently being exposed or the transmission is momentarily interrupted, waiting to receive the next frame (Note: This feature is not supported on some models of cameras, Camera this mark is equivalent to CAMERA_GET_IMAGE_PRIORITY_OLDEST)
            pRawData, frameHead = mvsdk.CameraGetImageBufferPriority(self.hcam, 1000, CAMERA_GET_IMAGE_PRIORITY_NEWEST)
            
            #pRawData, frameHead = mvsdk.CameraGetImageBuffer(self.hcam, 1000)
            mvsdk.CameraImageProcess(self.hcam, pRawData, self.frameBuffer, frameHead)
            mvsdk.CameraReleaseImageBuffer(self.hcam, pRawData)
                    
            frame_data = (mvsdk.c_ubyte * frameHead.uBytes).from_address(self.frameBuffer)
            frame = np.frombuffer(frame_data, dtype=np.uint8)
            frame = frame.reshape((frameHead.iHeight, frameHead.iWidth, 1 if frameHead.uiMediaType == mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3) )
            #filename = "%s.jpg"%str(time.time())
            #cv2.imwrite("./immagini_processo/" + filename, frame)
            #time.sleep(0.01)
            
            with Tlock:
                self.img_list.append([t,frame.copy()])
                #self.img_list.append([time.time(),cv2.flip(frame, 1)])
        

    def close(self):
        mvsdk.CameraUnInit(self.hcam)
        mvsdk.CameraAlignFree(self.frameBuffer)


@ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_int, ctypes.POINTER(mvsdk.tSdkFrameHead), ctypes.c_voidp)
#def GrabImageCallback(CameraHandle hCamera, BYTE *pFrameBuffer, tSdkFrameHead* pFrameHead,PVOID pContext):
def GrabImageCallback(hCamera, pRawData, pFrameHead, pContext):
    global Tlock, camera_instance
    print("new frame")
    with Tlock:
        #print("new image start")
        mvsdk.CameraImageProcess(camera_instance.hcam, pRawData, camera_instance.frameBuffer, pFrameHead.contents)
        #mvsdk.CameraReleaseImageBuffer(camera_instance.hcam, pRawData)
        
        frame_data = (mvsdk.c_ubyte * pFrameHead.contents.uBytes).from_address(camera_instance.frameBuffer)
        frame = np.frombuffer(frame_data, dtype=np.uint8)
        frame = frame.reshape((pFrameHead.contents.iHeight, pFrameHead.contents.iWidth, 1 if pFrameHead.contents.uiMediaType == mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3) )

        camera_instance.img_list.append([time.time(),frame])
        camera_instance.new_frame = True
        #cv2.imwrite("./img.png", frame)
        #print("new image end %sx%s"%(pFrameHead.contents.iWidth, pFrameHead.contents.iHeight))

    #sdk.CameraSetMediaType(camera_instance.hcam, pFrameBuffer, camera_instance.frameBuffer, pFrameHead)
    #CameraSaveImage(m_hCamera, strFileName.GetBuffer(), pImageData, pImageHead, FILE_BMP, 100);
        