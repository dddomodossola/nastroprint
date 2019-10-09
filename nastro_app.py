
# -*- coding: utf-8 -*-

import remi.gui as gui
from remi.gui import *
from remi import start, App
import nastro_mindvision
import widgets
import cv2
import threading
import time
import numpy as np
import image_utils
import glob #to list files
import shelve


class OpencvImageWidget(gui.Image):
    def __init__(self, filename, app_instance, **kwargs):
        self.app_instance = app_instance
        super(OpencvImageWidget, self).__init__("/%s/get_image_data" % id(self), **kwargs)
        self.img = cv2.imread(filename, 0)
        self.frame_index = 0
        self.set_image(filename)

    def set_image(self, filename):
        self.img = cv2.imread(filename, cv2.IMREAD_COLOR) #cv2.IMREAD_GRAYSCALE)#cv2.IMREAD_COLOR)
        self.update(self.app_instance)

    def update(self, app_instance):
        self.frame_index = self.frame_index + 1
        app_instance.execute_javascript("""
            url = '/%(id)s/get_image_data?index=%(frame_index)s';
            
            xhr = null;
            xhr = new XMLHttpRequest();
            xhr.open('GET', url, true);
            xhr.responseType = 'blob'
            xhr.onload = function(e){
                urlCreator = window.URL || window.webkitURL;
                urlCreator.revokeObjectURL(document.getElementById('%(id)s').src);
                imageUrl = urlCreator.createObjectURL(this.response);
                document.getElementById('%(id)s').src = imageUrl;
                //urlCreator.revokeObjectURL(imageUrl);
                //this = null;
            }
            xhr.send();
            """ % {'id': id(self), 'frame_index':self.frame_index})

    def refresh(self, opencv_img=None):
        self.img = opencv_img
        self.update(self.app_instance)

    def get_image_data(self, index=0):
        ret, jpeg = cv2.imencode('.png', self.img)
        if ret:
            headers = {'Content-type': 'image/png'}
            # tostring is an alias to tobytes, which wasn't added till numpy 1.9
            return [jpeg.tostring(), headers]
        return None, None


class MiniatureImageWidget(gui.HBox):
    def __init__(self, filename, *args, **kwargs):
        super(MiniatureImageWidget, self).__init__(*args, **kwargs)
        self.filename = filename
        self.style['margin'] = "2px"
        self.style['outline'] = "1px solid gray"
        self.check = gui.CheckBox(False)
        
        self.image = OpencvImageWidget(self.filename, height="100%")
        self.append([self.check, self.image])

    def delete(self):
        #add delete functionality
        pass


def process_thread(app,
                camera, 
                img_logo,
                stitch_minimum_overlap, 
                perspective_correction_value_horiz, 
                perspective_correction_value_vert):
    """ PRECONDIZIONE:
            -   NASTRO IN MOVIMENTO
        Prima fase di individuazione logo
            Misura lunghezza
                Prendi immagini
                ad ogni stitch (tramite l'intera larghezza immagine)
                    seziona l'intera immagine in fasce orizzontali
                    cerca tutte le occorrenze di queste fasce tramite match_template_all
                        se almeno 2 risultati abbiamo H logo
            Eventuale misura orizzontale logo, per individuare migliore zona di template matching al fine velocizzare di stitching
                Vedere tramite filtro verticale dove l'immagine cambia, dovrebbero essere individuabili N fasce verticali quanti sono i loghi
                    memorizzare aree e usarle per template matching
        Seconda fase 
            Scegliere un'immagine qualsiasi da usare come target iniziale centraggio immagine, 
            poi l'operatore può muovere in alto o in basso per centrare il logo
        Terza fase
            Stitching immagini
                Correggere prospettiva
                Tentare stitch, se fallisce 
                    allora stitchare in base a spostamento medio
                Se immagine stitch piu grande di logoX2 allora trovare l'immagine target iniziale e mostrare lo stitch centato al target
                    Ad ogni successivo stitch, eliminare eccesso H immagine che supera logoX2
    """
    app.process_thread_stop_flag = False

    
    #caricamento files di test
    """
    images_buffer = []
    image_files = glob.glob('./immagini_processo/frames/*.jpg')
    for img in image_files:
        img = cv2.imread(img, cv2.IMREAD_COLOR)
        images_buffer.append([time.time(), img])
    """
    images_buffer = None

    print("FASE 1")

    t = time.time()
    print("controllo numero immagini sufficienti")
    while app.process_thread_stop_flag==False and (time.time()-t)<5:
        if images_buffer!=None:
            if len(images_buffer)>3:
                break
        time.sleep(0.5)
        images_buffer = camera.get_image_buf()
    print(images_buffer)

    #creiamo scia loghi, in modo da individuare le aree occupate da ogni logo
    #img = image_utils.perspective_correction(images_buffer[0][1], perspective_correction_value_horiz, perspective_correction_value_vert)
    prev_thresh = image_utils.threshold(images_buffer[0][1])
    diff_images = prev_thresh.copy()#image_utils.threshold(images_buffer[0][1])#np.zeros((img.shape[1], img.shape[0]),np.uint8)
    diff_images[:] = 0
    kernel = np.ones((2,2),np.uint8)
    for t,img in images_buffer[:]:
        #comando stop operatore
        if app.process_thread_stop_flag:
            return
        #img = image_utils.perspective_correction(img, perspective_correction_value_horiz, perspective_correction_value_vert)
        img_thresh = image_utils.threshold(img)
        del img
        img_thresh = cv2.dilate(img_thresh,kernel,iterations = 1)
        diff_images = cv2.bitwise_or(diff_images, cv2.bitwise_xor(prev_thresh, img_thresh))

        prev_thresh = img_thresh

    diff_images = cv2.erode(diff_images,kernel,iterations = 1)
    diff_images = image_utils.perspective_correction(diff_images, perspective_correction_value_horiz, perspective_correction_value_vert)
    app.add_miniature(diff_images, "diff_images.png")

    """params = cv2.SimpleBlobDetector_Params()
    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = False
    # I loghi appaiono di colore bianco
    params.minThreshold = 100    # the graylevel of images
    params.maxThreshold = 255
    params.filterByColor = False
    #params.blobColor = 255
    # Filter by Area
    params.filterByArea = True
    params.minArea = 20
    detector = cv2.SimpleBlobDetector_create(params) #SimpleBlobDetector()
    # Detect blobs.
    keypoints = detector.detect(diff_images.astype(np.uint8))
    for k in keypoints:
        cv2.circle(img, (int(k.pt[0]), int(k.pt[1])), 20, (255,0,0), 5)
    """
    connectivity=1
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(diff_images , connectivity , cv2.CV_32S)
    lblareas = stats[:,cv2.CC_STAT_AREA]
    if len(stats) < 3:
        app.process_thread_stop_flag = True
        return
    ordered_regions_area = sorted(enumerate(lblareas), key=(lambda x: x[1]), reverse=True) #(index, area)
    print("ordered regions area %s"%str(ordered_regions_area))
    imax = ordered_regions_area[2][0] #1 skip the background indexed as 0
    #print("imax %s"%str(imax))
    #print(stats[imax])
    x1, y1, x2, y2 = (stats[imax, cv2.CC_STAT_LEFT], stats[imax, cv2.CC_STAT_TOP], stats[imax, cv2.CC_STAT_WIDTH]+stats[imax, cv2.CC_STAT_LEFT], stats[imax, cv2.CC_STAT_HEIGHT]+stats[imax, cv2.CC_STAT_TOP])
    
    img = image_utils.perspective_correction(images_buffer[0][1], perspective_correction_value_horiz, perspective_correction_value_vert)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 3)
    
    
    app.roi_logo_widget.style['left'] = gui.to_pix(x1)
    app.roi_logo_widget.style['top'] = gui.to_pix(y1)
    app.roi_logo_widget.style['width'] = gui.to_pix(x2-x1)
    app.roi_logo_widget.style['height'] = gui.to_pix(y2-y1)
    """
    imax = ordered_regions_area[2][0]
    x1, y1, x2, y2 = (stats[imax, cv2.CC_STAT_LEFT], stats[imax, cv2.CC_STAT_TOP], stats[imax, cv2.CC_STAT_WIDTH]+stats[imax, cv2.CC_STAT_LEFT], stats[imax, cv2.CC_STAT_HEIGHT]+stats[imax, cv2.CC_STAT_TOP])
    cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 3)
    
    imax = ordered_regions_area[3][0]
    x1, y1, x2, y2 = (stats[imax, cv2.CC_STAT_LEFT], stats[imax, cv2.CC_STAT_TOP], stats[imax, cv2.CC_STAT_WIDTH]+stats[imax, cv2.CC_STAT_LEFT], stats[imax, cv2.CC_STAT_HEIGHT]+stats[imax, cv2.CC_STAT_TOP])
    cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 3)
    """
    app.add_miniature(img, "loghi.png")
    #return

    roi_logo_x1 = x1
    roi_logo_x2 = x2

    vel_nastro = 0.0 #velocita' nastro utilizzata per lo stitching in caso di mancato pattern matching
    mem_t = 0 #tempo immagine precedente

    immagine_nastro = None
    vel_nastro_array = []
    similarity_array = []
    while not app.process_thread_stop_flag:
        #try:
        if True:    
            images_buffer = camera.get_image_buf()
            
            
            """
            print("attenzione, salvataggio immagini da disabilitare !!!")
            for t,img in images_buffer[:]: 
                filename = "%s.png"%str(time.time())
                cv2.imwrite("./immagini_processo/test/" + filename, img)
            """
            if not images_buffer is None:
                print("Thread loop - Immagini da giuntare: %s"%len(images_buffer))

                mem_t, img = images_buffer[0]
                immagine_nastro = image_utils.perspective_correction(img, perspective_correction_value_horiz, perspective_correction_value_vert)
                mem_image = img
                for t,img in images_buffer[1:]:
                    #img = image_utils.histogram_equalize(img)
                    #img = app.calibrate_image(img)
                    img = image_utils.perspective_correction(img, perspective_correction_value_horiz, perspective_correction_value_vert)
                    """
                    if offset_y==-1:
                        ok, _off_y, similarity_result = image_utils.find_stitch_offset(img, immagine_nastro, roi_logo_x, roi_logo_w)
                        if ok:
                            offset_y=_off_y
                        else:
                            immagine_nastro = img.copy()
                            index = 1
                            continue

                    #forse invertire le immagini top e bottom
                    #immagine_nastro = image_utils.stitch_fast(immagine_nastro, img, roi_logo_x, roi_logo_w)
                    immagine_nastro = image_utils.stitch_offset(img, immagine_nastro, offset_y*index)
                    index = index + 1
                    """
                    
                    #immagine_nastro = image_utils.stitch_fast(img, immagine_nastro, roi_logo_x, roi_logo_w)
                    ok, offset_y, similarity_result = image_utils.find_stitch_offset(img, mem_image, roi_logo_x1, roi_logo_x2, 30)
                    similarity_array.insert(0, similarity_result)
                    similarity_array = similarity_array[:30]
                    
                    #print("--------mean similarity: %s"%np.mean(similarity_array))
                    if len(similarity_array)<3 or similarity_result<np.mean(similarity_array): #ok
                        #print("offset y: %s"%offset_y)
                        immagine_nastro = image_utils.stitch_offset(img, immagine_nastro, offset_y)
                        #dato che il matching e' andato a buon fine, determiniamo la vel_nastro 
                        # in modo che se il prossimo matching non va' bene, stitchiamo in base a velocita'
                        vel_nastro_array.insert(0,float(offset_y)/(t-mem_t + 0.00000000001)) #sommo un numero molto piccolo per evitare DIV 0, e risparmiare un if
                        vel_nastro_array=vel_nastro_array[:30]
                        mem_t = t
                    else:
                        try:
                            
                            #continue
                            #stitch by previous offset_y relative to speed
                            
                            offset_y = int(np.mean(vel_nastro_array) * (t-mem_t))
                            #print(">>>>> vel nastro: %s    offset y: %s"%(np.mean(vel_nastro_array), offset_y))
                            
                            immagine_nastro = image_utils.stitch_offset(img, immagine_nastro, offset_y)
                            mem_t = t
                        except Exception as e:
                            print("Errore giunzione immagini a velocita'. offset_y:%s  -  err:%s"%(offset_y,e))
                    
                    h = int(app.spin_h_logo.get_value())
                    if not immagine_nastro is None:
                        if immagine_nastro.shape[0]>h:
                            break 

                    mem_image = img
                    
                if not immagine_nastro is None:
                    app.show_process_image(immagine_nastro)
                    #h = int(app.spin_h_logo.get_value()) #andrebbe determinato h logo
                    
                    #l'ultima immagine la rendo disponibile per il prossimo stitching
                    #img = image_utils.perspective_correction(img, perspective_correction_value_horiz, perspective_correction_value_vert)
                    #immagine_nastro = images_buffer[-1][1]#immagine_nastro[0:h,:]

                    #pulisco l'immagine per il successivo ciclo
                    immagine_nastro = None
                    #time.sleep(1.5)
        #except:
        #    print("exception")

def image_overprint_thread(app,
                camera,
                perspective_correction_value_horiz, 
                perspective_correction_value_vert):
    """ Si prende un'immagine
        tutte le successive immagini vengono matchate unite e sovrapposte a questa
    """
    app.process_thread_stop_flag = False
    
    images_buffer = None


    immagine_nastro = None
    vel_nastro_array = []
    similarity_array = []
    while not app.process_thread_stop_flag:
        try:
            images_buffer = camera.get_image_buf()
            
            """
            print("attenzione, salvataggio immagini da disabilitare !!!")
            for t,img in images_buffer[:]: 
                filename = "%s.png"%str(time.time())
                cv2.imwrite("./immagini_processo/test/" + filename, img)
            """
            if not images_buffer is None:
                print("Thread loop - Immagini da giuntare: %s"%len(images_buffer))

                mem_t, img = images_buffer[0]
                immagine_nastro = image_utils.perspective_correction(img, perspective_correction_value_horiz, perspective_correction_value_vert)
                mem_image = img
                for t,img in images_buffer[1:]:
                    #img = image_utils.histogram_equalize(img)
                    #img = app.calibrate_image(img)

                    #memorizzazione area logo da selezione a video
                    roi_logo_x1 = gui.from_pix(app.selection_area_widget.style['left'])
                    roi_logo_x2 = roi_logo_x1 + gui.from_pix(app.selection_area_widget.style['width'])

                    #tentativo stitching di nuova immagine in immagine_base e viceversa
                    # lo stitching on similarity piu' vicino a 0 (migliore) viene considerato
                    ok, offset_y, similarity_result = image_utils.find_stitch_offset(img, mem_image, roi_logo_x1, roi_logo_x2, 100)
                    ok, offset_y, similarity_result2 = image_utils.find_stitch_offset(mem_image, img, roi_logo_x1, roi_logo_x2, 100)
                    if similarity_result<similarity_result2:
                        #normale stitching di img in immagine nastro (dove Y img risulta piu' in basso)
                        immagine_nastro = image_utils.stitch_offset(img, immagine_nastro, offset_y)
                    else:
                        #stitching contrario immagine nastro in img (dove Y img risulta piu' alto)
                        # ma siccome vogliamo mantenere l'immagine nastro di sfondo, e volendo inoltre mantenere il risultato
                        # allineato in alto (immagine ferma), copiamo img in immagine_nastro, prendendo di img solo la parte che si sovrappone
                        # a immagine_nastro
                        immagine_nastro = image_utils.stitch_offset(img[offset_y:,:], immagine_nastro, offset_y)

                    #ritagliamo il risultato solo in caso la dimensione supera quella impostata sulla spinbox
                    h = int(app.spin_h_logo.get_value())
                    if immagine_nastro.shape[0] > h:
                        immagine_nastro = immagine_nastro[0:h,:)     # 0:mem_image.shape[1]] 
                    
                if not immagine_nastro is None:
                    app.show_process_image(immagine_nastro)
                    #h = int(app.spin_h_logo.get_value()) #andrebbe determinato h logo
                    
                    #l'ultima immagine la rendo disponibile per il prossimo stitching
                    #img = image_utils.perspective_correction(img, perspective_correction_value_horiz, perspective_correction_value_vert)
                    #immagine_nastro = images_buffer[-1][1]#immagine_nastro[0:h,:]

                    #pulisco l'immagine per il successivo ciclo
                    immagine_nastro = None
                    #time.sleep(1.5)
        except:
            print("exception")


def live_video_thread(app,
                camera, 
                perspective_correction_value_horiz, 
                perspective_correction_value_vert):

    app.process_thread_stop_flag = False
    images_buffer = None

    while not app.process_thread_stop_flag:
        images_buffer = camera.get_image_buf()

        if not images_buffer is None:
            print("Thread loop - Immagini da giuntare: %s"%len(images_buffer))

            app.show_process_image(images_buffer[0][1])
            continue
            for t,img in images_buffer[1:]:
                #img = image_utils.histogram_equalize(img)
                #img = app.calibrate_image(img)
                img = image_utils.perspective_correction(img, perspective_correction_value_horiz, perspective_correction_value_vert)

                app.show_process_image(img)
                

class MyApp(App):
    def __init__(self, *args, **kwargs):
        #DON'T MAKE CHANGES HERE, THIS METHOD GETS OVERWRITTEN WHEN SAVING IN THE EDITOR
        if not 'editing_mode' in kwargs.keys():
            super(MyApp, self).__init__(*args, static_file_path={'icons':'./icons/', 'immagini':'./immagini/'})

    def idle(self):
        #idle function called every update cycle
        if self.camera.continuous_acquire_run_flag:
            if not self.process_image is None:
                #correct brightness and contrast
                #cv2.addWeighted(self.process_image, alpha, np.zeros(img.shape, img.dtype),0, beta)
                cv2.addWeighted(self.process_image, contrast, np.zeros(self.process_image.shape, self.process_image.dtype),0, brightness)

                #cerca immagine di riferimento per centrare e tagliare immagine da mostrare
                self.camera_image.refresh(self.process_image)
                self.process_image = None
        self.resizer.update_position()
        self.dragger.update_position()
    
    def main(self):
        return MyApp.construct_ui(self)
        
    @staticmethod
    def construct_ui(self):
        cv2.setUseOptimized(True)
        #DON'T MAKE CHANGES HERE, THIS METHOD GETS OVERWRITTEN WHEN SAVING IN THE EDITOR
        global_container = VBox(height="100%")

        menu = gui.Menu(width='100%', height='30px')
        m1 = gui.MenuItem('File', width=100, height=30)
        m12 = gui.MenuItem('Open', width=100, height=30)
        m12.onclick.connect(self.menu_open_clicked)
        menu.append(m1)
        m1.append(m12)

        menubar = gui.MenuBar(width='100%', height='30px')
        menubar.style.update({"margin":"0px","position":"static","overflow":"visible","grid-area":"menubar","background-color":"#455eba", "z-index":"1"})
        menubar.append(menu)

        main_container = GridBox()
        main_container.attributes.update({"class":"GridBox","editor_constructor":"()","editor_varname":"main_container","editor_tag_type":"widget","editor_newclass":"False","editor_baseclass":"GridBox"})
        main_container.default_layout = """
        |container_commands |container_image                                                                                            |c_pars    |c_pars    |c_pars   |
        |container_commands |container_image                                                                                            |c_pars    |c_pars    |c_pars   |
        |container_commands |container_image                                                                                            |c_pars    |c_pars    |c_pars   |
        |container_commands |container_image                                                                                            |c_pars    |c_pars    |c_pars   |
        |container_commands |container_image                                                                                            |c_pars    |c_pars    |c_pars   |
        |container_commands |container_miniature                                                                                        |c_pars    |c_pars    |c_pars   |
        |container_process  |container_process                                                                                          |start_stop|live_video|overprint|    
        """
        main_container.set_from_asciiart(main_container.default_layout)

        #main_container.append(menubar,'menubar')
        container_commands = VBox()
        container_commands.style.update({"margin":"0px","display":"flex","justify-content":"flex-start","align-items":"center","flex-direction":"column","position":"static","overflow":"auto","grid-area":"container_commands","border-width":"2px","border-style":"dotted","border-color":"#8a8a8a"})

        #parametri di stile per i pulsanti
        btparams = {"margin":"2px","width":"100px","height":"30px","top":"20px","position":"static","overflow":"auto","order":"-1"}

        bt_trigger = Button('Trigger')
        bt_trigger.style.update({"background-color":"green", **btparams})

        container_commands.append(bt_trigger,'bt_trigger')

        bt_set_logo = Button('Imposta logo')
        bt_set_logo.style.update({"background-color":"darkorange", **btparams})
        container_commands.append(bt_set_logo,'bt_set_logo')
        main_container.append(container_commands,'container_commands')

        bt_search_logo = Button('Trova logo', style={"background-color":"orange", **btparams})
        bt_search_logo.onclick.connect(self.onclick_search_logo)
        container_commands.append(bt_search_logo, "bt_search_logo")

        bt_apply_calib_to_image = Button('Applica calibrazione', style={"background-color":"darkblue", **btparams})
        bt_apply_calib_to_image.onclick.connect(self.onclick_apply_calib_to_image)
        container_commands.append(bt_apply_calib_to_image, "bt_apply_calib_to_image")

        bt_perspective_correct = Button('Correggi prospettiva', style={"background-color":"darkblue", **btparams})
        bt_perspective_correct.onclick.connect(self.onclick_perspective_correction)
        container_commands.append(bt_perspective_correct, "bt_perspective_correct")

        bt_apply_filter = Button('Applica filtro threshold', style={"background-color":"black", **btparams})
        bt_apply_filter.onclick.connect(self.onclick_apply_filter, image_utils.threshold)
        container_commands.append(bt_apply_filter, "bt_apply_filter_threshold")

        bt_apply_filter = Button('Applica filtro canny', style={"background-color":"darkgray", **btparams})
        bt_apply_filter.onclick.connect(self.onclick_apply_filter, image_utils.canny)
        container_commands.append(bt_apply_filter, "bt_apply_filter_canny")

        bt_apply_filter = Button('Equalizza histogramma', style={"background-color":"violet", **btparams})
        bt_apply_filter.onclick.connect(self.onclick_apply_filter, image_utils.histogram_equalize)
        container_commands.append(bt_apply_filter, "bt_apply_filter_equalize")

        self.logo_image = None
        self.process_image = None #immagine impostata dal thread di processo, e aggiornata a video in idle()

        self.camera = nastro_mindvision.MindVisionCamera()
        
        #self.camera.start()

        self.camera_image = OpencvImageWidget("./test.bmp", self)#, width="100%", height="100%")
        self.camera_image.attributes.update({"class":"Widget","editor_constructor":"()","editor_varname":"container_image","editor_tag_type":"widget","editor_newclass":"False","editor_baseclass":"Widget"})
        container_camera_image = gui.Widget()
        container_camera_image.append(self.camera_image)

        self.resizer = widgets.ResizeHelper(container_camera_image, width=14, height=14, style={'background-color':'yellow', 'border':'1px dotted black', 'z-index':'2'})
        self.dragger = widgets.DragHelper(container_camera_image, width=14, height=14, style={'background-color':'yellow', 'border':'1px dotted black', 'z-index':'2'})
        self.selection_area_widget = gui.Widget(width=30, height=30, style={'background-color':'transparent', 'border':'2px dotted white', 'position':'absolute', 'top':'10px', 'left':'10px'})
        container_camera_image.append([self.resizer, self.dragger, self.selection_area_widget])
        self.resizer.setup(self.selection_area_widget, container_camera_image)
        self.dragger.setup(self.selection_area_widget, container_camera_image)
        main_container.append(container_camera_image,'container_image')
        container_camera_image.style.update({"margin":"0px","position":"relative","overflow":"scroll","grid-area":"container_image", "z-index": "-1"})

        #widget che mostra l'area di ricerca del logo
        self.roi_logo_widget = gui.Widget(width=100, height=100, style={'background-color':'transparent', 'border':'3px dotted green', 'position':'absolute', 'top':'10px', 'left':'10px'})
        container_camera_image.append(self.roi_logo_widget)

        container_parameters = VBox()
        container_parameters.style.update({"margin":"0px","display":"flex","justify-content":"flex-start","align-items":"center","flex-direction":"column","position":"static","overflow":"auto","grid-area":"c_pars","border-color":"#808080","border-width":"2px","border-style":"dotted"})
        
        lbl_exposure = Label('Esposizione')
        spin_exposure = SpinBox(1000,8,1000000,1)
        container = VBox(children = [lbl_exposure, spin_exposure], style = {"margin":"0px","display":"flex","justify-content":"flex-start","align-items":"center","flex-direction":"column","position":"static","overflow":"auto","order":"-1"})
        container_parameters.append(container, 'container_exposure')
        
        lbl_horizonal_perspective = Label('Horizontal perspective')
        self.spin_horizontal_perspective = gui.SpinBox(0, -1.0, 1.0, 0.1)
        lbl_vertical_perspective = Label('Vertical perspective')
        self.spin_vertical_perspective = gui.SpinBox(0, -1.0, 1.0, 0.1)
        self.spin_horizontal_perspective.onchange.connect(self.on_perspective_change)
        self.spin_vertical_perspective.onchange.connect(self.on_perspective_change)
        container = VBox(children = [lbl_horizonal_perspective, self.spin_horizontal_perspective, lbl_vertical_perspective, self.spin_vertical_perspective], style = {"margin":"0px","display":"flex","justify-content":"flex-start","align-items":"center","flex-direction":"column","position":"static","overflow":"auto","order":"-1"})
        container_parameters.append(container, 'container_perspective')

        
        lbl = Label("H Logo")
        self.spin_h_logo = SpinBox(1024, 100, 1000000,1)
        container = VBox(children = [lbl, self.spin_h_logo], style = {"margin":"0px","display":"flex","justify-content":"flex-start","align-items":"center","flex-direction":"column","position":"static","overflow":"auto","order":"-1"})
        container_parameters.append(container, 'container_h_logo')


        main_container.append(container_parameters,'c_pars')
        container_miniature = HBox()
        container_miniature.style.update({"margin":"0px","display":"flex","justify-content":"flex-end","align-items":"center","flex-direction":"row-reverse","position":"static","overflow-x":"scroll","grid-area":"container_miniature","background-color":"#e0e0e0"})
        main_container.append(container_miniature,'container_miniature')
        container_process = HBox()
        container_process.style.update({"margin":"0px","display":"flex","justify-content":"space-around","align-items":"center","flex-direction":"row","position":"static","overflow":"auto","grid-area":"container_process","background-color":"#e6e6e6","border-color":"#828282","border-width":"2px","border-style":"dotted"})
        main_container.append(container_process,'container_process')

        self.plot_histo_image_rgb = widgets.SvgPlot(256, 150)
        self.plot_histo_image_rgb.plot_red = widgets.SvgComposedPoly(0,0,256,1.0, 'red')
        self.plot_histo_image_rgb.plot_green = widgets.SvgComposedPoly(0,0,256,1.0, 'green')
        self.plot_histo_image_rgb.plot_blue = widgets.SvgComposedPoly(0,0,256,1.0, 'blue')
        self.plot_histo_image_rgb.append_poly([self.plot_histo_image_rgb.plot_red, self.plot_histo_image_rgb.plot_green, self.plot_histo_image_rgb.plot_blue])
        container_process.append(self.plot_histo_image_rgb)

        self.plot_histo_image_hsv = widgets.SvgPlot(256, 150)
        self.plot_histo_image_hsv.plot_hue = widgets.SvgComposedPoly(0,0,180,1.0, 'pink')
        self.plot_histo_image_hsv.plot_saturation = widgets.SvgComposedPoly(0,0,256,1.0, 'green')
        self.plot_histo_image_hsv.plot_value = widgets.SvgComposedPoly(0,0,256,1.0, 'black')
        self.plot_histo_image_hsv.append_poly([self.plot_histo_image_hsv.plot_hue, self.plot_histo_image_hsv.plot_saturation, self.plot_histo_image_hsv.plot_value])
        container_process.append(self.plot_histo_image_hsv)

        start_stop = Button('Start')
        start_stop.style.update({"margin":"0px","position":"static","overflow":"auto","background-color":"#39e600","font-weight":"bolder","font-size":"30px","height":"100%","letter-spacing":"3px"})
        main_container.append(start_stop,'start_stop')
        main_container.children['container_commands'].children['bt_trigger'].onclick.connect(self.onclick_bt_trigger)
        main_container.children['container_commands'].children['bt_set_logo'].onclick.connect(self.onclick_bt_set_logo)
        spin_exposure.onchange.connect(self.onchange_spin_exposure)
        main_container.children['start_stop'].onclick.connect(self.onclick_start_stop, self.camera.start)
        
        live_video = Button('Live Start')
        live_video.style.update({"margin":"0px","position":"static","overflow":"auto","background-color":"blue","font-weight":"bolder","font-size":"30px","height":"100%","letter-spacing":"3px"})
        main_container.append(live_video, "live_video")
        main_container.children['live_video'].onclick.connect(self.onclick_live_video, self.camera.start)
        
        image_overprint = Button('Overprint')
        image_overprint.style.update({"margin":"0px","position":"static","overflow":"auto","background-color":"violet","font-weight":"bolder","font-size":"30px","height":"100%","letter-spacing":"3px"})
        main_container.append(image_overprint, "overprint")
        main_container.children['overprint'].onclick.connect(self.onclick_overprint, self.camera.start)
        

        global_container.append([menubar, main_container])

        self.main_container = main_container

        #self.calibrate_camera()

        self.miniature_selection_list = []

        #load parameters
        self.shelve = shelve.open("./params.txt")
        try:
            print(self.shelve, self.shelve['spin_exposure'], self.shelve['spin_horizontal_perspective'])
            spin_exposure.set_value( int(self.shelve['spin_exposure']) )
            self.onchange_spin_exposure(spin_exposure, int(self.shelve['spin_exposure']) )

            self.spin_horizontal_perspective.set_value( float(self.shelve['spin_horizontal_perspective']) )
            self.spin_vertical_perspective.set_value( float(self.shelve['spin_vertical_perspective']) )

            self.spin_h_logo.set_value(float(self.shelve['spin_h_logo']))
        except:
            pass

        return global_container
    
    def onclose(self):
        self.shelve['spin_h_logo'] = self.spin_h_logo.get_value()
        self.shelve.close()
        self.camera.close()
        super(MyApp, self).onclose()

    def save_image_add_miniature(self, img, filename="", folder="./immagini_processo/"):
        if len(filename)<1:
            filename = "%s.png"%str(time.time())
        cv2.imwrite(folder + filename, img)
        miniature = MiniatureImageWidget("%s%s"%(folder, filename), height="100%", style={'margin-right':'2px'})
        miniature.image.onclick.connect(self.onclick_miniature, miniature)
        miniature.check.onchange.connect(self.onchange_miniature_selection, miniature)
        self.main_container.children['container_miniature'].append(miniature, filename)
        return miniature

    def onclick_bt_trigger(self, emitter):
        self.camera.set_trigger_mode(1) #software
        self.camera.trigger_software()
        #time.sleep(1)
        t, image = self.camera.img_list[-1] #.get_last_image()
        #image = self.calibrate_image(image)
        #perspective_correction_value_horiz = float(self.spin_horizontal_perspective.get_value())
        #perspective_correction_value_vert = float(self.spin_vertical_perspective.get_value())
        #image = image_utils.perspective_correction(image, perspective_correction_value_horiz, perspective_correction_value_vert)
        self.save_image_add_miniature(image)
        self.camera_image.refresh(image)

    def onclick_bt_set_logo(self, emitter):
        x = gui.from_pix(self.selection_area_widget.style['left'])
        y = gui.from_pix(self.selection_area_widget.style['top'])
        w = gui.from_pix(self.selection_area_widget.style['width'])
        h = gui.from_pix(self.selection_area_widget.style['height'])

        self.roi_logo_widget.style['left'] = self.selection_area_widget.style['left']
        self.roi_logo_widget.style['top'] = "1px" #self.selection_area_widget.style['top']
        self.roi_logo_widget.style['width'] = self.selection_area_widget.style['width']
        self.roi_logo_widget.style['height'] = gui.to_pix(self.camera_image.img.shape[0])#self.selection_area_widget.style['height']

        img = self.camera_image.img[y:y+h, x:x+w]
        self.logo_image = img
        self.save_image_add_miniature(img, "logo.png")
        
    def onclick_search_logo(self, emitter):
        h, w, channels = self.logo_image.shape
        ok, x, y, similarity_result = image_utils.match_template(self.logo_image, self.camera_image.img, similarity=0.1)
        if ok: #la marca e' stata trovata
            self.selection_area_widget.style['left'] = gui.to_pix(x)
            self.selection_area_widget.style['top'] = gui.to_pix(y)
            self.selection_area_widget.style['width'] = gui.to_pix(w)
            self.selection_area_widget.style['height'] = gui.to_pix(h)

    def onclick_apply_calib_to_image(self, emitter):
        img = self.calibrate_image(self.camera_image.img)
        self.camera_image.refresh(img)

    def onclick_apply_filter(self, emitter, method):
        img = method(self.camera_image.img)
        self.camera_image.refresh(img)

    def onchange_spin_exposure(self, emitter, value):
        self.shelve['spin_exposure'] = int(value)
        self.camera.set_exposure(int(value))

    def on_perspective_change(self, emitter, value):
        self.onclick_perspective_correction()

    def onclick_perspective_correction(self, emitter=None):
        w = float(self.spin_horizontal_perspective.get_value())
        h = float(self.spin_vertical_perspective.get_value())

        self.shelve['spin_horizontal_perspective'] = w
        self.shelve['spin_vertical_perspective'] = h

        #test
        #self.camera_image.set_image("./test.bmp")
        result = image_utils.perspective_correction(self.camera_image.img, w, h)
        self.camera_image.refresh(result)

    def onclick_miniature(self, emitter, miniature):
        self.camera_image.refresh(miniature.image.img)

        #https://docs.opencv.org/2.4/doc/tutorials/imgproc/histograms/histogram_comparison/histogram_comparison.html
        #Use the function compareHist to get a numerical parameter that express how well two histograms match with each other.
        img = miniature.image.img
        r,g,b = image_utils.get_histograms_rgb(img)
        scale_y = self.plot_histo_image_rgb.height/max(max(r), max(g), max(b))
        self.plot_histo_image_rgb.plot_red.scale(1, scale_y)
        self.plot_histo_image_rgb.plot_green.scale(1, scale_y)
        self.plot_histo_image_rgb.plot_blue.scale(1, scale_y)
        i = 0
        for v in r:
            self.plot_histo_image_rgb.plot_red.add_coord(i, v)
            i = i+1

        i = 0
        for v in g:
            self.plot_histo_image_rgb.plot_green.add_coord(i, v)
            i = i+1

        i = 0
        for v in b:
            self.plot_histo_image_rgb.plot_blue.add_coord(i, v)
            i = i+1
        self.plot_histo_image_rgb.render()

        h,s,value = image_utils.get_histograms_hsv(img)
        scale_y = self.plot_histo_image_hsv.height/max(max(h), max(s), max(value))
        self.plot_histo_image_hsv.plot_hue.scale(1, scale_y)
        self.plot_histo_image_hsv.plot_saturation.scale(1, scale_y)
        self.plot_histo_image_hsv.plot_value.scale(1, scale_y)
        i = 0
        for v in h:
            self.plot_histo_image_hsv.plot_hue.add_coord(i, v)
            i = i+1

        i = 0
        for v in s:
            self.plot_histo_image_hsv.plot_saturation.add_coord(i, v)
            i = i+1

        i = 0
        for v in value:
            self.plot_histo_image_hsv.plot_value.add_coord(i, v)
            i = i+1
        self.plot_histo_image_hsv.render()

    def onchange_miniature_selection(self, emitter, selected, miniature):
        print(selected)
        if selected:
            self.miniature_selection_list.append(miniature)
        else:
            self.miniature_selection_list.remove(miniature)
        if len(self.miniature_selection_list) > 1:
            print("stitching %s images"%len(self.miniature_selection_list))
            result = self.miniature_selection_list[0].image.img.copy()

            roi_logo_x = gui.from_pix(self.roi_logo_widget.style['left'])
            roi_logo_w = gui.from_pix(self.roi_logo_widget.style['width'])
            t = time.time()

            w = float(self.spin_horizontal_perspective.get_value())
            h = float(self.spin_vertical_perspective.get_value())

            #result = self.calibrate_image(result)
            result = image_utils.perspective_correction(result, w, h)
            for m in self.miniature_selection_list[1:]:
                img = m.image.img#self.calibrate_image(m.image.img)
                img = image_utils.perspective_correction(img, w, h)
                result = image_utils.stitch_fast(result, img, roi_logo_x, roi_logo_x+roi_logo_w, 30.0) #image_utils.stitch(result, m.image.img, 20.0)
            print(time.time() - t)
            self.camera_image.refresh(result)

    def onclick_start_stop(self, emitter, method):
        method() #start or stop
        if not (method == self.camera.stop):
            w = float(self.spin_horizontal_perspective.get_value())
            h = float(self.spin_vertical_perspective.get_value())
            self.camera.set_trigger_mode(1) 
            self.process_thread = threading.Thread(target = process_thread, args = [self, self.camera, self.logo_image, 15, w, h] )
            self.process_thread.setDaemon(True)
            self.process_thread.start()
            emitter.set_text("Stop")
            emitter.style['background-color'] = 'red'
            self.main_container.children['start_stop'].onclick.connect(self.onclick_start_stop, self.camera.stop)
            main_container.set_from_asciiart("""
            |container_image                                                                        | c_pars  |
            |container_image                                                                        | c_pars  |
            |container_image                                                                        | c_pars  |
            |container_image                                                                        | c_pars  |
            |container_image                                                                        | c_pars  |
            |container_image                                                                        | c_pars  |
            |container_image                                                                        | c_pars  |
            |container_image                                                                        | start_stop            |
            """)
            
        else:
            self.process_thread_stop_flag = True
            emitter.set_text("Start")
            emitter.style['background-color'] = 'green'
            self.main_container.children['start_stop'].onclick.connect(self.onclick_start_stop, self.camera.start)
            main_container.set_from_asciiart(self.main_container.default_layout)

    def onclick_live_video(self, emitter, method):
        method() #start or stop
        if not (method == self.camera.stop):
            w = float(self.spin_horizontal_perspective.get_value())
            h = float(self.spin_vertical_perspective.get_value())
            self.camera.set_trigger_mode(1) 
            self.process_thread = threading.Thread(target = live_video_thread, args = [self, self.camera, w, h] )
            self.process_thread.setDaemon(True)
            self.process_thread.start()
            emitter.set_text("Live Stop")
            emitter.style['background-color'] = 'yellow'
            self.main_container.children['live_video'].onclick.connect(self.onclick_live_video, self.camera.stop)
            main_container.set_from_asciiart("""
            |container_image                                                                        | c_pars  |
            |container_image                                                                        | c_pars  |
            |container_image                                                                        | c_pars  |
            |container_image                                                                        | c_pars  |
            |container_image                                                                        | c_pars  |
            |container_image                                                                        | c_pars  |
            |container_image                                                                        | c_pars  |
            |container_image                                                                        | live_video            |
            """)

        else:
            self.process_thread_stop_flag = True
            emitter.set_text("Live Start")
            emitter.style['background-color'] = 'blue'
            self.main_container.children['live_video'].onclick.connect(self.onclick_live_video, self.camera.start)
            main_container.set_from_asciiart(self.main_container.default_layout)

    def onclick_overprint(self, emitter, method):
        method() #start or stop
        if not (method == self.camera.stop):
            w = float(self.spin_horizontal_perspective.get_value())
            h = float(self.spin_vertical_perspective.get_value())
            self.camera.set_trigger_mode(1) 
            self.process_thread = threading.Thread(target = image_overprint_thread, args = [self, self.camera, w, h] )
            self.process_thread.setDaemon(True)
            self.process_thread.start()
            emitter.set_text("Stop")
            emitter.style['background-color'] = 'red'
            self.main_container.children['overprint'].onclick.connect(self.onclick_overprint, self.camera.stop)
            main_container.set_from_asciiart("""
            |container_image                                                                        | c_pars  |
            |container_image                                                                        | c_pars  |
            |container_image                                                                        | c_pars  |
            |container_image                                                                        | c_pars  |
            |container_image                                                                        | c_pars  |
            |container_image                                                                        | c_pars  |
            |container_image                                                                        | c_pars  |
            |container_image                                                                        | overprint             |
            """)

        else:
            self.process_thread_stop_flag = True
            emitter.set_text("Overprint")
            emitter.style['background-color'] = 'violet'
            self.main_container.children['overprint'].onclick.connect(self.onclick_overprint, self.camera.start)
            main_container.set_from_asciiart(self.main_container.default_layout)

    def menu_open_clicked(self, widget):
        self.fileselectionDialog = gui.FileSelectionDialog('File Selection Dialog', 'Select files and folders', True, '.')
        self.fileselectionDialog.confirm_value.connect(self.on_fileselection_dialog_confirm)
        # here is returned the Input Dialog widget, and it will be shown
        self.fileselectionDialog.show(self)

    def on_fileselection_dialog_confirm(self, widget, filelist):
        # a list() of filenames and folders is returned
        print('Selected files: %s' % ','.join(filelist))
        #if len(filelist):
        for f in filelist:
            #f = filelist[0]
            self.camera_image.set_image(f)

            miniature = MiniatureImageWidget(f, height="100%", style={'margin-right':'2px'})
            miniature.image.onclick.connect(self.onclick_miniature, miniature)
            miniature.check.onchange.connect(self.onchange_miniature_selection, miniature)
            self.main_container.children['container_miniature'].append(miniature)

    def calibrate_camera(self):
        #load all calib images
        #store calibration data in a variable
        #apply calibration to images before stitching 
        
        images = glob.glob('./immagini/*.jpg')
        image_points, obj_points = image_utils.get_caltab_points(images, self.camera_image, 7, 6)
        self.camera_image.set_image(images[0])
        gray = cv2.cvtColor(self.camera_image.img,cv2.COLOR_BGR2GRAY)
        ret, self.camera_calib_mtx, self.camera_calib_dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, image_points, gray.shape[::-1],None,None)

        h,  w = self.camera_image.img.shape[:2]
        self.camera_calib_newcameramtx, self.camera_calib_roi=cv2.getOptimalNewCameraMatrix(self.camera_calib_mtx,self.camera_calib_dist,(w,h),1,(w,h))

        # undistort
        result = self.calibrate_image(self.camera_image.img)
        self.camera_image.refresh(result)

        """
        mean_error = 0
        for i in xrange(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
            tot_error += error

        print "total error: ", mean_error/len(objpoints)
        """

    def calibrate_image(self, img):
        # undistort
        result = cv2.undistort(img, self.camera_calib_mtx, self.camera_calib_dist, None, self.camera_calib_newcameramtx)

        h,  w = img.shape[:2]
        # crop the image
        x,y,w,h = self.camera_calib_roi
        return result[y:y+h, x:x+w]

    def show_process_image(self, image):
        print("show_process_image")
        self.process_image = image.copy()

    def add_miniature(self, img, filename=""):
        with self.update_lock:
            self.save_image_add_miniature(img, filename)


#Configuration
configuration = {'config_project_name': 'MyApp', 'config_address': '0.0.0.0', 'config_port': 8081, 'config_multiple_instance': False, 'config_enable_file_cache': True, 'config_start_browser': True, 'config_resourcepath': './res/'}

if __name__ == "__main__":
    # start(MyApp,address='127.0.0.1', port=8081, multiple_instance=False,enable_file_cache=True, update_interval=0.1, start_browser=True)
    start(MyApp, address=configuration['config_address'], port=configuration['config_port'], 
                        multiple_instance=configuration['config_multiple_instance'], 
                        enable_file_cache=configuration['config_enable_file_cache'],
                        start_browser=configuration['config_start_browser'], update_interval=0.05, title= "Nastro")
