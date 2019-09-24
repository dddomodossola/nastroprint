import remi.gui as gui
import remi.server
import collections

class DraggableItem(gui.EventSource):
    def __init__(self, container, **kwargs):
        gui.EventSource.__init__(self)
        self.container = container
        self.refWidget = None
        self.parent = None
        self.active = False
        self.origin_x = -1
        self.origin_y = -1
        self.snap_grid_size = 1

    def setup(self, refWidget, newParent):
        #refWidget is the target widget that will be resized
        #newParent is the container
        if self.parent:
            try:
                self.parent.remove_child(self)
            except:
                pass
        if newParent==None:
            return
        self.parent = newParent
        self.refWidget = refWidget
        
        try:
            self.parent.append(self)
        except:
            pass
        self.update_position()
            
    def start_drag(self, emitter, x, y):
        self.active = True
        self.container.onmousemove.connect(self.on_drag)
        self.container.onmouseup.connect(self.stop_drag)
        self.container.onmouseleave.connect(self.stop_drag, 0, 0)
        self.origin_x = -1
        self.origin_y = -1

    @gui.decorate_event
    def stop_drag(self, emitter, x, y):
        self.active = False
        self.update_position()
        return ()

    def on_drag(self, emitter, x, y):
        pass

    def update_position(self):
        pass

    def set_snap_grid_size(self, value):
        self.snap_grid_size = value
    
    def round_grid(self, value):
        return int(value/self.snap_grid_size)*self.snap_grid_size


class ResizeHelper(gui.Widget, DraggableItem):

    def __init__(self, container, **kwargs):
        super(ResizeHelper, self).__init__(**kwargs)
        DraggableItem.__init__(self, container, **kwargs)
        self.style['float'] = 'none'
        self.style['position'] = 'absolute'
        self.style['left']='0px'
        self.style['top']='0px'
        self.onmousedown.connect(self.start_drag)

    def setup(self, refWidget, newParent):
        if type(refWidget) in [gui.Widget, gui.Button, gui.GridBox, gui.VBox, gui.HBox, 
                                gui.ListView, gui.DropDown, gui.Label, gui.Image, gui.Link,
                                gui.TableWidget, gui.TextInput, gui.CheckBox, gui.CheckBox, 
                                gui.CheckBoxLabel, gui.Slider, gui.SpinBox, gui.ColorPicker,
                                gui.Svg, gui.VideoPlayer, gui.Progress]:
            DraggableItem.setup(self, refWidget, newParent)

    def on_drag(self, emitter, x, y):
        if self.active:
            if self.origin_x == -1:
                self.origin_x = float(x)
                self.origin_y = float(y)
                self.refWidget_origin_w = gui.from_pix(self.refWidget.style['width'])
                self.refWidget_origin_h = gui.from_pix(self.refWidget.style['height'])
            else:
                self.refWidget.style['width'] = gui.to_pix( self.round_grid( self.refWidget_origin_w + float(x) - self.origin_x ) )
                self.refWidget.style['height'] = gui.to_pix( self.round_grid( self.refWidget_origin_h + float(y) - self.origin_y ) )
                self.update_position()

    def update_position(self):
        self.style['position']='absolute'
        if self.refWidget:
            if 'left' in self.refWidget.style and 'top' in self.refWidget.style:
                self.style['left']=gui.to_pix(gui.from_pix(self.refWidget.style['left']) + gui.from_pix(self.refWidget.style['width']) )
                self.style['top']=gui.to_pix(gui.from_pix(self.refWidget.style['top']) + gui.from_pix(self.refWidget.style['height']) )


class DragHelper(gui.Widget, DraggableItem):

    def __init__(self, container, **kwargs):
        super(DragHelper, self).__init__(**kwargs)
        DraggableItem.__init__(self, container, **kwargs)
        self.style['float'] = 'none'
        self.style['position'] = 'absolute'
        self.style['left']='0px'
        self.style['top']='0px'
        self.onmousedown.connect(self.start_drag)

    def setup(self, refWidget, newParent):
        if type(refWidget) in [gui.Widget, gui.Button, gui.GridBox, gui.VBox, gui.HBox, 
                                gui.ListView, gui.DropDown, gui.Label, gui.Image, gui.Link,
                                gui.TableWidget, gui.TextInput, gui.CheckBox, gui.CheckBox, 
                                gui.CheckBoxLabel, gui.Slider, gui.SpinBox, gui.ColorPicker,
                                gui.Svg, gui.VideoPlayer, gui.Progress]:
            DraggableItem.setup(self, refWidget, newParent)

    def on_drag(self, emitter, x, y):
        if self.active:
            if self.origin_x == -1:
                self.origin_x = float(x)
                self.origin_y = float(y)
                self.refWidget_origin_x = gui.from_pix(self.refWidget.style['left'])
                self.refWidget_origin_y = gui.from_pix(self.refWidget.style['top'])
            else:
                self.refWidget.style['left'] = gui.to_pix( self.round_grid( self.refWidget_origin_x + float(x) - self.origin_x ) )
                self.refWidget.style['top'] = gui.to_pix( self.round_grid( self.refWidget_origin_y + float(y) - self.origin_y ) )
                self.update_position()

    def update_position(self):
        self.style['position']='absolute'
        if self.refWidget:
            if 'left' in self.refWidget.style and 'top' in self.refWidget.style:
                self.style['left']=gui.to_pix(gui.from_pix(self.refWidget.style['left'])-gui.from_pix(self.style['width']))
                self.style['top']=gui.to_pix(gui.from_pix(self.refWidget.style['top'])-gui.from_pix(self.style['width']))


class SvgComposedPoly(gui.SvgGroup):
    """ A group of polyline and circles
    """
    def __init__(self, x, y, maxlen, stroke, color, **kwargs):
        super(SvgComposedPoly, self).__init__(x, y, **kwargs)
        self.maxlen = maxlen
        self.plotData = gui.SvgPolyline(self.maxlen)
        self.append(self.plotData)
        self.set_stroke(stroke, color)
        self.set_fill(color)
        self.circle_radius = stroke
        self.circles_list = list()
        self.x_factor = 1.0
        self.y_factor = 1.0

    def add_coord(self, x, y):
        """ Adds a coord to the polyline and creates another circle
        """
        x = x*self.x_factor
        y = y*self.y_factor
        self.plotData.add_coord(x, y)
        #self.circles_list.append(gui.SvgCircle(x, y, self.circle_radius))
        #self.append(self.circles_list[-1])
        #if len(self.circles_list) > self.maxlen:
        #    self.remove_child(self.circles_list[0])
        #    del self.circles_list[0]

    def scale(self, x_factor, y_factor):
        self.x_factor = x_factor/self.x_factor
        self.y_factor = y_factor/self.y_factor
        self.plotData.attributes['points'] = "" 
        tmpx = collections.deque()
        tmpy = collections.deque()

        for c in self.circles_list:
            self.remove_child(c)
        self.circles_list = list()

        while len(self.plotData.coordsX)>0:
            tmpx.append( self.plotData.coordsX.popleft() )
            tmpy.append( self.plotData.coordsY.popleft() )

        while len(tmpx)>0:
            self.add_coord(tmpx.popleft(), tmpy.popleft())
            
        self.x_factor = x_factor
        self.y_factor = y_factor
        

class SvgPlot(gui.Svg):
    def __init__(self, width, height):
        super(SvgPlot, self).__init__(width, height)
        self.width = width
        self.height = height
        self.polyList = []
        self.font_size = 15
        self.plot_inner_border = self.font_size
        self.textYMin = gui.SvgText(0, self.height + self.font_size, "min")
        self.textYMax = gui.SvgText(0, 0, "max")
        self.textYMin.style['font-size'] = gui.to_pix(self.font_size)
        self.textYMax.style['font-size'] = gui.to_pix(self.font_size)
        self.append([self.textYMin, self.textYMax])

    def append_poly(self, polys):
        for poly in polys:
            self.append(poly)
            self.polyList.append(poly)
            poly.textXMin = gui.SvgText(0, 0, "actualValue")
            poly.textXMax = gui.SvgText(0, 0, "actualValue")
            poly.textYVal = gui.SvgText(0, 0, "actualValue")
            poly.textYVal.style['font-size'] = gui.to_pix(self.font_size)

            poly.lineYValIndicator = gui.SvgLine(0, 0, 0, 0)
            poly.lineXMinIndicator = gui.SvgLine(0, 0, 0, 0)
            poly.lineXMaxIndicator = gui.SvgLine(0, 0, 0, 0)
            self.append([poly.textXMin, poly.textXMax, poly.textYVal, poly.lineYValIndicator, 
                poly.lineXMinIndicator, poly.lineXMaxIndicator])

    def remove_poly(self, poly):
        self.remove_child(poly)
        self.polyList.remove(poly)
        self.remove_child(poly.textXMin)
        self.remove_child(poly.textXMax)
        self.remove_child(poly.textYVal)

    def render(self):
        self.set_viewbox(-self.plot_inner_border, -self.plot_inner_border, self.width + self.plot_inner_border * 2,
                         self.height + self.plot_inner_border * 2)
        if len(self.polyList) < 1:
            return
        minX = min(self.polyList[0].plotData.coordsX)
        maxX = max(self.polyList[0].plotData.coordsX)
        minY = min(self.polyList[0].plotData.coordsY)
        maxY = max(self.polyList[0].plotData.coordsY)

        for poly in self.polyList:
            minX = min(minX, min(poly.plotData.coordsX))
            maxX = max(maxX, max(poly.plotData.coordsX))
            minY = min(minY, min(poly.plotData.coordsY))
            maxY = max(maxY, max(poly.plotData.coordsY))
        self.textYMin.set_text("min:%s" % minY)
        self.textYMax.set_text("max:%s" % maxY)

        i = 1
        for poly in self.polyList:
            scaledTranslatedYpos = (-poly.plotData.coordsY[-1] + maxY + (self.height-(maxY-minY))/2.0)

            textXpos = self.height / (len(self.polyList) + 1) * i

            poly.textXMin.set_text(str(min(poly.plotData.coordsX)))
            poly.textXMin.set_fill(poly.attributes['stroke'])

            poly.textXMin.set_position(-textXpos, (min(poly.plotData.coordsX) - minX) )
            poly.textXMin.attributes['transform'] = "rotate(%s)" % (-90)
            poly.textXMax.set_text(str(max(poly.plotData.coordsX)))
            poly.textXMax.set_fill(poly.attributes['stroke'])
            poly.textXMax.set_position(-textXpos, (max(poly.plotData.coordsX) - minX) )

            poly.textXMax.attributes['transform'] = "rotate(%s)" % (-90)
            poly.textYVal.set_text(str(poly.plotData.coordsY[-1]))
            poly.textYVal.set_fill(poly.attributes['stroke'])
            poly.textYVal.set_position(0, scaledTranslatedYpos)

            poly.lineYValIndicator.set_stroke(1, poly.attributes['stroke'])
            poly.lineXMinIndicator.set_stroke(1, poly.attributes['stroke'])
            poly.lineXMaxIndicator.set_stroke(1, poly.attributes['stroke'])
            poly.lineYValIndicator.set_coords(0, scaledTranslatedYpos, self.width, scaledTranslatedYpos)
            poly.lineXMinIndicator.set_coords((min(poly.plotData.coordsX) - minX), 0,
                                              (min(poly.plotData.coordsX) - minX), self.height)
            poly.lineXMaxIndicator.set_coords((max(poly.plotData.coordsX) - minX), 0,
                                              (max(poly.plotData.coordsX) - minX), self.height)
            poly.attributes['transform'] = ('translate(%s,%s)' % (-minX, maxY + (self.height-(maxY-minY))/2.0) + 
                                            ' scale(%s,%s)' % ((1.0), -(1.0)))
            i = i + 1
