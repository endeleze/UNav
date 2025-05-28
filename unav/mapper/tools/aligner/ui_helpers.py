import tkinter as tk
from tkinter import ttk
import warnings
from PIL import Image, ImageTk
import math

class AutoScrollbar(ttk.Scrollbar):
    """
    A scrollbar that hides itself when not needed.
    Only works with the grid geometry manager.
    """

    def set(self, lo, hi):
        # Hide scrollbar if the entire content is visible
        if float(lo) <= 0.0 and float(hi) >= 1.0:
            self.grid_remove()
        else:
            self.grid()
            super().set(lo, hi)

    def pack(self, **kwargs):
        raise tk.TclError(f"Cannot use pack with {self.__class__.__name__}")

    def place(self, **kwargs):
        raise tk.TclError(f"Cannot use place with {self.__class__.__name__}")

class CanvasImage:
    """
    Canvas widget for displaying, zooming, and panning large images with optional scrollbars.
    Handles very large images using a pyramid for performance.
    """

    def __init__(self, placeholder, path):
        """
        Initialize a CanvasImage widget.

        Args:
            placeholder: Parent Tk widget.
            path (str): Path to the image file.
        """
        self.imscale = 1.0  # Current image scale (zoom)
        self.__delta = 1.3  # Zoom step per wheel event
        self.__filter = Image.Resampling.LANCZOS 
        self.__previous_state = 0  # Previous keyboard state
        self.path = path

        # Frame to contain canvas and scrollbars
        self.__imframe = ttk.Frame(placeholder)

        # Create horizontal and vertical auto-scrollbars
        hbar = AutoScrollbar(self.__imframe, orient='horizontal')
        vbar = AutoScrollbar(self.__imframe, orient='vertical')
        hbar.grid(row=1, column=0, sticky='we')
        vbar.grid(row=0, column=1, sticky='ns')

        # Create the canvas and bind it to the scrollbars
        self.canvas = tk.Canvas(self.__imframe, highlightthickness=0,
                                xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        self.canvas.grid(row=0, column=0, sticky='nswe')
        self.canvas.update()

        hbar.configure(command=self.__scroll_x)
        vbar.configure(command=self.__scroll_y)

        # Bind canvas and mouse events
        self.canvas.bind('<Configure>', lambda event: self.__show_image())
        self.canvas.bind('<ButtonPress-1>', self.__move_from)
        self.canvas.bind('<B1-Motion>',     self.__move_to)
        self.canvas.bind('<MouseWheel>',    self.__wheel)
        self.canvas.bind('<Button-5>',      self.__wheel)
        self.canvas.bind('<Button-4>',      self.__wheel)
        self.canvas.bind('<Key>', lambda event: self.canvas.after_idle(self.__keystroke, event))

        # Set up large image support and pyramid
        self.__huge = False
        self.__huge_size = 14000
        self.__band_width = 1024
        Image.MAX_IMAGE_PIXELS = 1000000000
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.__image = Image.open(self.path)
        self.imwidth, self.imheight = self.__image.size

        if self.imwidth * self.imheight > self.__huge_size * self.__huge_size \
           and self.__image.tile[0][0] == 'raw':
            self.__huge = True
            self.__offset = self.__image.tile[0][2]
            self.__tile = [
                self.__image.tile[0][0],
                [0, 0, self.imwidth, 0],
                self.__offset,
                self.__image.tile[0][3]
            ]
        self.__min_side = min(self.imwidth, self.imheight)

        # Create image pyramid for efficient zooming
        self.__pyramid = [self.smaller()] if self.__huge else [Image.open(self.path)]
        self.__ratio = max(self.imwidth, self.imheight) / self.__huge_size if self.__huge else 1.0
        self.__curr_img = 0
        self.__scale = self.imscale * self.__ratio
        self.__reduction = 2
        w, h = self.__pyramid[-1].size
        while w > 512 and h > 512:
            w /= self.__reduction
            h /= self.__reduction
            self.__pyramid.append(self.__pyramid[-1].resize((int(w), int(h)), self.__filter))
        # Add the image as a rectangle container for proper image mapping
        self.container = self.canvas.create_rectangle((0, 0, self.imwidth, self.imheight), width=0)
        self.__show_image()
        self.canvas.focus_set()

    def smaller(self):
        """
        Resize image proportionally and return a smaller version (used for huge images).
        """
        w1, h1 = float(self.imwidth), float(self.imheight)
        w2, h2 = float(self.__huge_size), float(self.__huge_size)
        aspect_ratio1 = w1 / h1
        aspect_ratio2 = w2 / h2
        if aspect_ratio1 == aspect_ratio2:
            image = Image.new('RGB', (int(w2), int(h2)))
            k = h2 / h1
            w = int(w2)
        elif aspect_ratio1 > aspect_ratio2:
            image = Image.new('RGB', (int(w2), int(w2 / aspect_ratio1)))
            k = h2 / w1
            w = int(w2)
        else:
            image = Image.new('RGB', (int(h2 * aspect_ratio1), int(h2)))
            k = h2 / h1
            w = int(h2 * aspect_ratio1)
        i, j, n = 0, 1, round(0.5 + self.imheight / self.__band_width)
        while i < self.imheight:
            print(f'\rOpening image: {j} from {n}', end='')
            band = min(self.__band_width, self.imheight - i)
            self.__tile[1][3] = band
            self.__tile[2] = self.__offset + self.imwidth * i * 3
            self.__image.close()
            self.__image = Image.open(self.path)
            self.__image.size = (self.imwidth, band)
            self.__image.tile = [self.__tile]
            cropped = self.__image.crop((0, 0, self.imwidth, band))
            image.paste(cropped.resize((w, int(band * k) + 1), self.__filter), (0, int(i * k)))
            i += band
            j += 1
        print('\r' + 30 * ' ' + '\r', end='')
        return image

    def redraw_figures(self):
        """
        Override in child classes to redraw figures when zooming or panning.
        """
        pass

    def grid(self, **kw):
        """
        Place the CanvasImage widget on the parent using grid.
        """
        self.__imframe.grid(**kw)
        self.__imframe.grid(sticky='nswe')
        self.__imframe.rowconfigure(0, weight=1)
        self.__imframe.columnconfigure(0, weight=1)

    def pack(self, **kw):
        raise Exception('Cannot use pack with the widget ' + self.__class__.__name__)

    def place(self, **kw):
        raise Exception('Cannot use place with the widget ' + self.__class__.__name__)

    def __scroll_x(self, *args, **kwargs):
        self.canvas.xview(*args)
        self.__show_image()

    def __scroll_y(self, *args, **kwargs):
        self.canvas.yview(*args)
        self.__show_image()

    def __show_image(self):
        """
        Show the image on the canvas. Implements correct image zoom and efficient scrolling.
        """
        box_image = self.canvas.coords(self.container)
        box_canvas = (
            self.canvas.canvasx(0),
            self.canvas.canvasy(0),
            self.canvas.canvasx(self.canvas.winfo_width()),
            self.canvas.canvasy(self.canvas.winfo_height()))
        box_img_int = tuple(map(int, box_image))
        box_scroll = [
            min(box_img_int[0], box_canvas[0]), min(box_img_int[1], box_canvas[1]),
            max(box_img_int[2], box_canvas[2]), max(box_img_int[3], box_canvas[3])]
        if box_scroll[0] == box_canvas[0] and box_scroll[2] == box_canvas[2]:
            box_scroll[0] = box_img_int[0]
            box_scroll[2] = box_img_int[2]
        if box_scroll[1] == box_canvas[1] and box_scroll[3] == box_canvas[3]:
            box_scroll[1] = box_img_int[1]
            box_scroll[3] = box_img_int[3]
        self.canvas.configure(scrollregion=tuple(map(int, box_scroll)))
        x1 = max(box_canvas[0] - box_image[0], 0)
        y1 = max(box_canvas[1] - box_image[1], 0)
        x2 = min(box_canvas[2], box_image[2]) - box_image[0]
        y2 = min(box_canvas[3], box_image[3]) - box_image[1]
        if int(x2 - x1) > 0 and int(y2 - y1) > 0:
            if self.__huge and self.__curr_img < 0:
                h = int((y2 - y1) / self.imscale)
                self.__tile[1][3] = h
                self.__tile[2] = self.__offset + self.imwidth * int(y1 / self.imscale) * 3
                self.__image.close()
                self.__image = Image.open(self.path)
                self.__image.size = (self.imwidth, h)
                self.__image.tile = [self.__tile]
                image = self.__image.crop((int(x1 / self.imscale), 0, int(x2 / self.imscale), h))
            else:
                image = self.__pyramid[max(0, self.__curr_img)].crop(
                    (int(x1 / self.__scale), int(y1 / self.__scale),
                     int(x2 / self.__scale), int(y2 / self.__scale)))
            imagetk = ImageTk.PhotoImage(image.resize((int(x2 - x1), int(y2 - y1)), self.__filter))
            imageid = self.canvas.create_image(
                max(box_canvas[0], box_img_int[0]),
                max(box_canvas[1], box_img_int[1]),
                anchor='nw', image=imagetk)
            self.canvas.lower(imageid)
            self.canvas.imagetk = imagetk

    def __move_from(self, event):
        self.canvas.scan_mark(event.x, event.y)

    def __move_to(self, event):
        self.canvas.scan_dragto(event.x, event.y, gain=1)
        self.__show_image()

    def outside(self, x, y):
        bbox = self.canvas.coords(self.container)
        if bbox[0] < x < bbox[2] and bbox[1] < y < bbox[3]:
            return False
        else:
            return True

    def __wheel(self, event):
        """
        Zoom with mouse wheel.
        """
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        if self.outside(x, y):
            return
        scale = 1.0
        if event.num == 5 or event.delta == -120:
            if round(self.__min_side * self.imscale) < 30:
                return
            self.imscale /= self.__delta
            scale        /= self.__delta
        if event.num == 4 or event.delta == 120:
            i = min(self.canvas.winfo_width(), self.canvas.winfo_height()) >> 1
            if i < self.imscale:
                return
            self.imscale *= self.__delta
            scale        *= self.__delta
        k = self.imscale * self.__ratio
        self.__curr_img = min(
            (-1) * int(math.log(k, self.__reduction)),
            len(self.__pyramid) - 1
        )
        self.__scale = k * math.pow(self.__reduction, max(0, self.__curr_img))
        self.canvas.scale('all', x, y, scale, scale)
        self.redraw_figures()
        self.__show_image()

    def __keystroke(self, event):
        """
        Allow keyboard navigation (WASD, arrows, numpad).
        """
        if event.state - self.__previous_state == 4:
            pass
        else:
            self.__previous_state = event.state
            if event.keycode in [68, 39, 102]:
                self.__scroll_x('scroll',  1, 'unit', event=event)
            elif event.keycode in [65, 37, 100]:
                self.__scroll_x('scroll', -1, 'unit', event=event)
            elif event.keycode in [87, 38, 104]:
                self.__scroll_y('scroll', -1, 'unit', event=event)
            elif event.keycode in [83, 40, 98]:
                self.__scroll_y('scroll',  1, 'unit', event=event)

    def crop(self, bbox):
        """
        Crop a rectangle from the image.

        Args:
            bbox: (left, upper, right, lower) tuple.
        """
        if self.__huge:
            band = bbox[3] - bbox[1]
            self.__tile[1][3] = band
            self.__tile[2] = self.__offset + self.imwidth * bbox[1] * 3
            self.__image.close()
            self.__image = Image.open(self.path)
            self.__image.size = (self.imwidth, band)
            self.__image.tile = [self.__tile]
            return self.__image.crop((bbox[0], 0, bbox[2], band))
        else:
            return self.__pyramid[0].crop(bbox)

    def destroy(self):
        """
        Destructor for CanvasImage.
        Closes all resources and destroys the frame.
        """
        self.__image.close()
        map(lambda i: i.close, self.__pyramid)
        del self.__pyramid[:]
        del self.__pyramid
        self.canvas.destroy()
        self.__imframe.destroy()
