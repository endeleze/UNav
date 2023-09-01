import tkinter as tk
from tkinter import *
import numpy as np
from tkinter import ttk,filedialog
from tkinter.messagebox import showinfo
from ttkbootstrap import Style

import cv2
from PIL import Image, ImageTk, ImageDraw
import logging

from track import Hloc
from navigation import Trajectory,actions,command_debug, command_count
from visualization.destination_selection import Destination_window

import argparse
from os.path import dirname,join,exists,realpath
from os import listdir
import yaml
import loader

from tqdm import tqdm

class Main_window(ttk.Frame):
    image_types=['.jpg', '.png', '.jpeg', '.JPG', '.PNG','.HEIC']
    def __init__(self,master, map_data=None,hloc=None,trajectory=None,**config):
        ttk.Frame.__init__(self, master=master)
        self.config=config
        self.map_data=map_data
        self.map_scale=self.config['location']['scale']
        self.keyframe_name=map_data['keyframe_name']
        self.database_loc=map_data['database_loc']
        self.keyframe_location=map_data['keyframe_location']
        self.destination_list_name,self.destination_list_location=[],[]
        anchor_name=map_data['anchor_name']
        anchor_location=map_data['anchor_location']
        ind=0
        while anchor_name[ind][0]!='w':
            self.destination_list_name.append(anchor_name[ind])
            self.destination_list_location.append(anchor_location[ind])
            ind+=1
        self.hloc=hloc
        self.trajectory=trajectory
        self.destination=[]
        self.initial=False
        self.halfway = False
        self.GT=None
        self.retrieval=True
        self.__layout_design()
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)

    def __layout_design(self):
        windowWidth = self.master.winfo_reqwidth()
        windowHeight = self.master.winfo_reqheight()
        self.positionRight = int(self.master.winfo_screenwidth() / 2 - windowWidth / 2)
        self.positionDown = int(self.master.winfo_screenheight() / 2 - windowHeight / 2)
        self.master.geometry("+{}+{}".format(self.positionRight, self.positionDown))
        self.master.title('Visualization')
        self.pack(side="left", fill="both", expand=False)
        self.master.geometry('2000x1200')
        self.master.columnconfigure(1, weight=1)
        self.master.columnconfigure(3, pad=7)
        self.master.rowconfigure(3, weight=1)
        self.master.rowconfigure(6, pad=7)
        """
        Localization image list
        """

        lbl = Label(self, text="Query frames:")
        lbl.grid(row=0, column=0, sticky=W, pady=4, ipadx=2)

        self.testing_image_folder=join(self.config['IO_root'],self.config['testing_image_folder'])

        self.query_names=sorted([file for file in listdir(self.testing_image_folder) if file.endswith('.png')])

        var2 = tk.StringVar()
        self.lb = tk.Listbox(self, listvariable=var2)

        self.scrollbar = Scrollbar(self, orient=VERTICAL)
        self.lb.config(yscrollcommand=self.scrollbar.set)
        self.scrollbar.config(command=self.lb.yview)

        self.lb.bind('<Double-Button-1>', lambda event, action='double':self.animate(action))
        self.lb.bind('<Up>', lambda event, action='up':
        self.animate(action))
        self.lb.bind('<Down>', lambda event, action='down':
        self.animate(action))
        self.testing_image_list = [i.split('.')[0] for i in sorted(self.query_names)]
        for i in self.testing_image_list:
            self.lb.insert('end', i)

        self.scrollbar.grid(row=7, column=1, columnspan=2, rowspan=4, padx=2, sticky='sn')

        """
        Function area
        """

        self.lb.grid(row=7, column=0, columnspan=1, rowspan=4, padx=2,
                     sticky=E + W + S + N)

        self.c=ttk.LabelFrame(self, text='Retrieval animation')
        self.c.grid(row=6, column=0, columnspan=2, padx=2,sticky=E + W)
        self.v1 = tk.IntVar()
        self.retrieval=True
        self.lb2 = tk.Radiobutton(self.c,
                                  text='Trun on',
                                  command=self.__retrieval_setting,
                                  variable=self.v1,
                                  value=0).pack(side=LEFT)
        self.lb2 = tk.Radiobutton(self.c,
                                  text='Trun off',
                                  command=self.__retrieval_setting,
                                  variable=self.v1,
                                  value=1).pack(side=RIGHT)
        self.shift = tk.Frame(self)
        self.shift.grid(row=14, column=0, padx=10, pady=1, sticky='we')
        scale = Label(self.shift, text='Retrieval number:')
        scale.pack(side=LEFT)
        self.e1 = tk.Entry(self.shift, width=3, justify='left')
        self.e1.pack(side=LEFT)
        self.e1.insert(END, '10')
        self.shift1 = tk.Frame(self)
        self.shift1.grid(row=12, column=0, padx=10, pady=1, sticky='we')
        scale = Label(self.shift1, text='Plotting scale:')
        scale.pack(side=LEFT)
        self.e2 = tk.Entry(self.shift1, width=5, justify='right')
        self.e2.insert(END, '0.1')
        self.e2.pack(side=LEFT)
        scale = Label(self.shift1, text='\'/pixel')
        scale.pack(side=LEFT)
        # ---------------------------------------------------
        separatorv1 = ttk.Separator(self, orient='vertical')
        separatorv1.grid(row=0, column=3, padx=10, columnspan=1, rowspan=70, sticky="sn")
        ebtn = tk.Button(self, text='Save Animation', width=16, command=self.gif_generator)
        ebtn.grid(row=15, column=0, padx=10, columnspan=1, rowspan=1)
        ebtn = tk.Button(self, text='Help', width=16, command=self.__help_info)
        ebtn.grid(row=16, column=0, padx=10, columnspan=1, rowspan=1)
        ebtn = tk.Button(self, text='Clear Destination', width=16, command=self.__clear_destination)
        ebtn.grid(row=17, column=0, padx=10, columnspan=1, rowspan=1)

        """
        Result area
        """

        location_config=self.config['location']
        floorplan_path=join(self.config['IO_root'],'data',location_config['place'],location_config['building'],location_config['floor'])

        fl_selected = filedialog.askopenfilename(initialdir=floorplan_path, title='Select Floor Plan')
        floorplan = Image.open(fl_selected)
        floorplan_draw = ImageDraw.Draw(floorplan)
        self.floorplan=floorplan.copy()
        
        for x,y in self.keyframe_location:
            floorplan_draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(0, 255, 0))

        self.floorplan_with_keyframe = floorplan.copy()
        width, height = floorplan.size
        self.plot_scale =width/3400
        scale = 1600 / width
        newsize = (1600, int(height * scale))
        floorplan = floorplan.resize(newsize)
        tkimage1 = ImageTk.PhotoImage(floorplan)
        self.myvar1 = Label(self, image=tkimage1)
        self.myvar1.image = tkimage1
        self.myvar1.grid(row=0, column=4, columnspan=1, rowspan=40, sticky="snew")
        self.myvar1.bind('<Double-Button-1>', lambda event, action='double':
                            self.select_destination(action))
        
        self.length = 0

    def __clear_destination(self):
        self.destination=[]
        self.set_destination()

    def __retrieval_setting(self):
        if self.v1.get()==0:
            self.retrieval=True
        else:
            self.retrieval=False

    def prepare_image(self,image):
        draw=ImageDraw.Draw(image)
        l=60*self.plot_scale
        x_,y_=50*self.plot_scale,l
        ang=0
        x1, y1 = x_ - 40*self.plot_scale * np.sin(ang), y_ - 40*self.plot_scale * np.cos(ang)
        draw.ellipse((x_ - 20*self.plot_scale, y_ - 20*self.plot_scale, x_ + 20*self.plot_scale, y_ + 20*self.plot_scale), fill=(50, 0, 106))
        draw.line([(x_, y_), (x1, y1)], fill=(50, 0, 106), width=int(10*self.plot_scale))
        im=np.array(image)
        im=cv2.putText(im, 'Estimation pose', (int(100*self.plot_scale), int(l)), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 0), round(2*self.plot_scale), cv2.LINE_AA)
        image=Image.fromarray(im)
        draw = ImageDraw.Draw(image)
        if self.GT:
            l+=70*self.plot_scale
            x_, y_ = 50*self.plot_scale, l
            x1, y1 = x_ - 40*self.plot_scale * np.sin(ang), y_ - 40*self.plot_scale * np.cos(ang)
            draw.ellipse((x_ - 20*self.plot_scale, y_ - 20*self.plot_scale, x_ + 20*self.plot_scale, y_ + 20*self.plot_scale), fill=(255, 0, 255))
            draw.line([(x_, y_), (x1, y1)], fill=(255, 0, 255), width=int(10*self.plot_scale))
            im = np.array(image)
            im = cv2.putText(im, 'Ground truth pose', (int(100*self.plot_scale), int(l)), cv2.FONT_HERSHEY_SIMPLEX,
                             1, (0, 0, 0), round(2**self.plot_scale), cv2.LINE_AA)
            image = Image.fromarray(im)
            draw = ImageDraw.Draw(image)
        if self.retrieval:
            l+=70*self.plot_scale
            x_, y_ = 50*self.plot_scale, l
            x1, y1 = x_ - 20*self.plot_scale * np.sin(ang), y_ - 20 *self.plot_scale* np.cos(ang)
            draw.ellipse((x_ - 10*self.plot_scale, y_ - 10*self.plot_scale, x_ + 10*self.plot_scale, y_ + 10*self.plot_scale), fill=(255, 0, 0))
            draw.line([(x_, y_), (x1, y1)], fill=(255, 0, 0), width=int(7*self.plot_scale))
            im = np.array(image)
            im = cv2.putText(im, 'Similar images', (int(100*self.plot_scale), int(l)), cv2.FONT_HERSHEY_SIMPLEX,
                             1, (0, 0, 0), round(2*self.plot_scale), cv2.LINE_AA)
            image = Image.fromarray(im)
            draw = ImageDraw.Draw(image)
        if len(self.destination)>0:
            l+=70*self.plot_scale
            vertices = self.__star_vertices([50*self.plot_scale, l], 30)
            draw.polygon(vertices, fill='red')
            im = np.array(image)
            im = cv2.putText(im, 'Destination', (int(100*self.plot_scale), int(l)), cv2.FONT_HERSHEY_SIMPLEX,
                             1, (0, 0, 0), round(2*self.plot_scale), cv2.LINE_AA)
            image = Image.fromarray(im)
            draw = ImageDraw.Draw(image)
        draw.rectangle([(10,5),(400+100*self.plot_scale,l+40*self.plot_scale)],outline='black',width=int(2*self.plot_scale))
        return draw,image
    
    def action(self,rot_ang,distance,floorplan):
        floorplan_numpy=np.array(floorplan)
        h,_,_=floorplan_numpy.shape
        rot_ang=(-rot_ang)%360
        rot_clock = round(rot_ang / 30) % 12
        floorplan_numpy = cv2.putText(floorplan_numpy, u'Please walk %.1f meters along %d clock\n' % (distance * self.scale, rot_clock), (10, h - 80), cv2.FONT_HERSHEY_SIMPLEX,
                         1.2, (255, 0, 0), 2, cv2.LINE_AA)
        print('Please walk %.1f meters along %d clock\n' % (distance * self.scale, rot_clock))
        return floorplan_numpy
    
    def gif(self):
        c,d=int(self.e4.get()),int(self.e5.get())
        dic_path = join(self.testing_image_folder, self.query_names[0])
        image = Image.open(dic_path)
        b, _ = image.size
        scale = c / b
        
        floorplan = self.floorplan.copy()
        draw_floorplan, floorplan = self.prepare_image(floorplan)
        e, _ = floorplan.size
        gif_shrink = e / d
        gif_ims=[]
        navigation_paths=[]

        for dic in tqdm(self.query_names,desc='Generate frames'):
            floorplan = self.floorplan.copy()
            draw_floorplan, floorplan = self.prepare_image(floorplan)
            dic_path = join(self.testing_image_folder, dic)
            image = Image.open(dic_path)
            query_image=image.copy()
            query_image_numpy=np.array(query_image)
            query_image_numpy=cv2.cvtColor(query_image_numpy,cv2.COLOR_BGR2RGB)
            pose = self.hloc.get_location(query_image_numpy)
            if pose:
                navigation_paths.append(pose[:2])
                if len(navigation_paths)>1:
                    x0,y0=navigation_paths[0]
                    for x1,y1 in navigation_paths[1:]:
                        draw_floorplan.line([(x1, y1), (x0, y0)], fill=(255, 0, 0), width=15)
                        x0,y0=x1,y1
                self.scale = float(self.e2.get())
                floorplan_numpy = np.array(floorplan)
                h, _, _ = floorplan_numpy.shape
                floorplan_numpy = cv2.putText(floorplan_numpy, 'Current location:  [%d,%d],  orientation:  %d degree' % (
                    pose[0], pose[1], pose[2]), (10, h - 140), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 0, 255), 2, cv2.LINE_AA)
                if len(self.destination)>0:
                    for d in self.destination:
                        xx,yy=self.keyframe_location[self.keyframe_name.index(d)]
                        floorplan_numpy = cv2.putText(floorplan_numpy, 'Destination location:  [%d,%d]' % (
                            xx, yy), (10, h - 200), cv2.FONT_HERSHEY_SIMPLEX,
                                        1, (0, 0, 0), 2, cv2.LINE_AA)
                floorplan = Image.fromarray(floorplan_numpy)
                draw_floorplan = ImageDraw.Draw(floorplan)
                if len(self.destination)>0:
                    paths=[]
                    for d in self.destination:
                        xx,yy=self.keyframe_location[self.keyframe_name.index(d)]
                        vertices = self.__star_vertices([xx, yy], 30)
                        draw_floorplan.polygon(vertices, fill='red', outline='red')
                    for destination_id in self.destination:
                        path_list=self.trajectory.calculate_path(pose[:2], destination_id)
                        if len(path_list)>0:
                            paths+=path_list

                    paths=[pose[:2]]+paths
                    for i in range(1,len(paths)):
                        x0, y0=paths[i-1]
                        x1, y1=paths[i]
                        vertices = self.__star_vertices([x0, y0], 15)
                        draw_floorplan.polygon(vertices, fill='yellow', outline='red')
                        draw_floorplan.line([(x0, y0), (x1, y1)], fill=(0,255,0), width=int(5*self.plot_scale))

                if self.retrieval:
                    for i in self.hloc.retrived_image_index:
                        x_,y_,ang=self.database_loc[i]
                        x1, y1 = x_ - 20*self.plot_scale * np.sin(ang), y_ - 20*self.plot_scale * np.cos(ang)
                        draw_floorplan.ellipse((x_ - 10*self.plot_scale, y_ - 10*self.plot_scale, x_ + 10*self.plot_scale, y_ + 10*self.plot_scale), fill=(255, 0, 0))
                        draw_floorplan.line([(x_, y_), (x1, y1)], fill=(255, 0, 0), width=int(7*self.plot_scale))
                draw_query = ImageDraw.Draw(query_image)
                for p in self.hloc.matched_2D:
                    xx, yy = p[0]-0.5, p[1]-0.5
                    draw_query.rectangle([(xx - 10, yy - 10), (xx + 10, yy + 10)], fill=(0, 0, 255))

                width_query, height_query = query_image.size
                newsize = (int(width_query * scale), int(height_query * scale))
                query_image = query_image.resize(newsize)
                draw_floorplan = ImageDraw.Draw(floorplan)
                matched_3D=np.array(self.hloc.matched_3D)
                self.matched_3D=np.ones((3,matched_3D.shape[0]))
                self.matched_3D[0,:]=matched_3D[:,0]
                self.matched_3D[1, :] = matched_3D[:, 2]
                matched_3D=(self.hloc.T@self.matched_3D).T
                for points in matched_3D:
                    x, y = points
                    draw_floorplan.rectangle([(x - 3, y - 3), (x + 3, y + 3)], fill=(0, 0, 255))

                width_query, height_query = query_image.size
                width_floorplan, height_floorplan = floorplan.size
                floorplan.paste(query_image, (width_floorplan - width_query, height_floorplan - height_query))

                for i, p in enumerate(self.hloc.matched_2D):
                    x2, y2 = (p[0]-0.5) * scale + width_floorplan - width_query, (p[1]-0.5) * scale + height_floorplan - height_query
                    x3, y3 = matched_3D[i]
                    draw_floorplan.line([(x3, y3), (x2, y2)], fill=(0, 255, 255), width=1)

                x1, y1 = pose[0] - 80 * np.sin(pose[2]/180*np.pi), pose[1] - 80 * np.cos(pose[2]/180*np.pi)
                draw_floorplan.ellipse((pose[0] - 40, pose[1] - 40, pose[0] + 40, pose[1] + 40), fill=(50, 0, 106))
                draw_floorplan.line([(pose[0], pose[1]), (x1, y1)], fill=(50, 0, 106), width=20)
                newsize = (int(width_floorplan / gif_shrink), int(height_floorplan / gif_shrink))
            else:
                floorplan = self.floorplan.copy()
                draw_floorplan, floorplan = self.prepare_image(floorplan)

                width_query, height_query = query_image.size
                newsize = (int(width_query * scale), int(height_query * scale))
                query_image = query_image.resize(newsize)

                width_query, height_query = query_image.size
                width_floorplan, height_floorplan = floorplan.size
                floorplan.paste(query_image, (width_floorplan - width_query, height_floorplan - height_query))
                width_floorplan, height_floorplan = floorplan.size
                newsize = (int(width_floorplan / gif_shrink), int(height_floorplan / gif_shrink))

            gif_ims.append(floorplan.resize(newsize))

        outf=str(self.e6.get())
        output_filename=join(self.config['IO_root'],self.config['testing_image_folder'], outf+'.mp4')
        out = cv2.VideoWriter(output_filename, 
                      cv2.VideoWriter_fourcc(*'mp4v'), 
                      float(self.e1.get()), 
                      newsize)
        
        
        for im in gif_ims:
            cv2_im = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
            out.write(cv2_im)

        out.release()

        self.win.destroy()
        showinfo("Saved!", "Animation saved to:\n{}".format(output_filename))

    def gif_generator(self):
        self.win = tk.Toplevel(self.master)
        self.win.wm_title("Warning!!!")
        self.win.geometry('330x130')
        self.win.geometry("+{}+{}".format(self.positionRight, self.positionDown))
        l = tk.Label(self.win, text="Save animation to outpath?")
        l.grid(row=1, column=0, columnspan=4, rowspan=2,
               padx=20, pady=30)

        b1 = ttk.Button(self.win, text="Yes", width=6, command=self.gif_config)
        b1.grid(row=3, column=0, columnspan=1, padx=40, rowspan=3)

        b2 = ttk.Button(self.win, text="No", width=6, command=self.win.destroy)
        b2.grid(row=3, column=1, columnspan=1, padx=40, rowspan=3)
    
    def gif_config(self):
        self.win.destroy()
        self.win = tk.Toplevel(self.master)
        self.win.wm_title("Animation setting")
        self.win.geometry('400x200')
        self.win.geometry("+{}+{}".format(self.positionRight, self.positionDown))
        self.shift = tk.Frame(self.win)
        self.shift.grid(row=0, column=0)
        self.shift1 = tk.Frame(self.shift)
        self.shift1.grid(row=0, column=0, pady=5, padx=60)
        scale = Label(self.shift1, text='Frame width in Gif:')
        scale.pack(side=LEFT)
        self.e4 = tk.Entry(self.shift1, width=4, justify='center')
        self.e4.insert(END, '1500')
        self.e4.pack(side=LEFT)
        self.shift2 = tk.Frame(self.shift)
        self.shift2.grid(row=1, column=0, pady=5, padx=60)
        scale = Label(self.shift2, text='Output gif width:')
        scale.pack(side=LEFT)
        self.e5 = tk.Entry(self.shift2, width=4, justify='center')
        self.e5.insert(END, '900')
        self.e5.pack(side=LEFT)
        self.shift4 = tk.Frame(self.shift)
        self.shift4.grid(row=2, column=0, pady=5, padx=60)
        scale = Label(self.shift4, text='fps:')
        scale.pack(side=LEFT)
        self.e1 = tk.Entry(self.shift4, width=2, justify='center')
        self.e1.insert(END, '1')
        self.e1.pack(side=LEFT)

        self.shift5 = tk.Frame(self.shift)
        self.shift5.grid(row=3, column=0, pady=5, padx=60)
        scale = Label(self.shift5, text='Gif name:')
        scale.pack(side=LEFT)
        self.e6 = tk.Entry(self.shift5, width=15, justify='center')
        # self.e6.insert(END, self.config['testing_image_folder'])
        self.e6.pack(side=LEFT)
        scale = Label(self.shift5, text='.mp4')
        scale.pack(side=LEFT)

        self.shift3 = tk.Frame(self.shift)
        self.shift3.grid(row=4, column=0, pady=5)
        b1 = ttk.Button(self.win, text="Go", width=6, command=self.gif)
        b1.grid(row=4, column=0, columnspan=1, padx=30, rowspan=1, sticky='w')
        b2 = ttk.Button(self.win, text="Cancel", width=6, command=self.win.destroy)
        b2.grid(row=4, column=0, columnspan=1, padx=20, rowspan=1, sticky='e')

    def __help_info(self):
        self.info = tk.Toplevel(self.master)
        self.info.geometry('800x650')
        self.info.title('Instruction')
        self.info.geometry("+{}+{}".format(self.positionRight - 300, self.positionDown - 200))
        # This will create a LabelFrame
        label_frame1 = LabelFrame(self.info, height=100, text='Steps')
        label_frame1.pack(expand='yes', fill='both')

        label1 = Label(label_frame1, text='1. Double click floor plan to pick a destination point.')
        label1.place(x=0, y=5)

        label2 = Label(label_frame1, text='2. Choose a query in query list to see localization.')
        label2.place(x=0, y=35)

        label3 = Label(label_frame1,
                       text='3. Click and see retrieved images and double click pairs to check local matches.')
        label3.place(x=0, y=65)

        label4 = Label(label_frame1, text='4. Repeat above to see other query results.')
        label4.place(x=0, y=95)

        label_frame1 = LabelFrame(self.info, height=400, text='Buttons')
        label_frame1.pack(expand='yes', fill='both', side='bottom')

        label_1 = LabelFrame(label_frame1, height=60, text='Retrieval animation')
        label_1.place(x=5, y=23)
        label1 = Label(label_1, text='Turn on or turn off the display of retrieval images')
        label1.pack()

        label_2 = LabelFrame(label_frame1, height=40, text='Plotting scale')
        label_2.place(x=5, y=70)
        label2 = Label(label_2, text='How many foot of per pixel.')
        label2.pack()

        label_3 = LabelFrame(label_frame1, height=40, text='Retrieval number')
        label_3.place(x=5, y=117)
        label3 = Label(label_3,
                       text='How many retrieved images to use.')
        label3.pack()

        label_4 = LabelFrame(label_frame1, height=60, text='Save Animation')
        label_4.place(x=5, y=164)
        label4 = Label(label_4,
                       text='Save the localization results.')
        label4.pack()
    
    def __star_vertices(self,center,r):
        out_vertex = [(r*self.plot_scale * np.cos(2 * np.pi * k / 5 + np.pi / 2- np.pi / 5) + center[0],
                       r*self.plot_scale * np.sin(2 * np.pi * k / 5 + np.pi / 2- np.pi / 5) + center[1]) for k in range(5)]
        r = r/2
        in_vertex = [(r*self.plot_scale * np.cos(2 * np.pi * k / 5 + np.pi / 2 ) + center[0],
                      r*self.plot_scale * np.sin(2 * np.pi * k / 5 + np.pi / 2 ) + center[1]) for k in range(5)]
        vertices = []
        for i in range(5):
            vertices.append(out_vertex[i])
            vertices.append(in_vertex[i])
        vertices = tuple(vertices)
        return vertices
    
    def __visualize_result(self,floorplan_with_keyframe):
        draw_floorplan_with_keyframe=ImageDraw.Draw(floorplan_with_keyframe)
        l=60*self.plot_scale
        x_,y_=50*self.plot_scale,l
        ang=0
        x1, y1 = x_ - 40*self.plot_scale * np.sin(ang), y_ - 40*self.plot_scale * np.cos(ang)
        draw_floorplan_with_keyframe.ellipse((x_ - 20*self.plot_scale, y_ - 20*self.plot_scale, x_ + 20*self.plot_scale, y_ + 20*self.plot_scale), fill=(50, 0, 106))
        draw_floorplan_with_keyframe.line([(x_, y_), (x1, y1)], fill=(50, 0, 106), width=int(10*self.plot_scale))
        floorplan_with_keyframe_np=np.array(floorplan_with_keyframe)
        h, _, _ = floorplan_with_keyframe_np.shape 
        floorplan_with_keyframe_np=cv2.putText(floorplan_with_keyframe_np, 'Estimation pose', (int(100*self.plot_scale), int(l)), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 0), round(2*self.plot_scale), cv2.LINE_AA)
        floorplan_with_keyframe=Image.fromarray(floorplan_with_keyframe_np)
        draw_floorplan_with_keyframe = ImageDraw.Draw(floorplan_with_keyframe)
        if self.GT:
            l+=70*self.plot_scale
            x_, y_ = 50*self.plot_scale, l
            x1, y1 = x_ - 40*self.plot_scale * np.sin(ang), y_ - 40*self.plot_scale * np.cos(ang)
            draw_floorplan_with_keyframe.ellipse((x_ - 20*self.plot_scale, y_ - 20*self.plot_scale, x_ + 20*self.plot_scale, y_ + 20*self.plot_scale), fill=(255, 0, 255))
            draw_floorplan_with_keyframe.line([(x_, y_), (x1, y1)], fill=(255, 0, 255), width=int(10*self.plot_scale))
            floorplan_with_keyframe_np = np.array(floorplan_with_keyframe)
            floorplan_with_keyframe_np = cv2.putText(floorplan_with_keyframe_np, 'Ground truth pose', (int(100*self.plot_scale), int(l)), cv2.FONT_HERSHEY_SIMPLEX,
                             1, (0, 0, 0), round(2**self.plot_scale), cv2.LINE_AA)
            floorplan_with_keyframe = Image.fromarray(floorplan_with_keyframe_np)
            draw_floorplan_with_keyframe = ImageDraw.Draw(floorplan_with_keyframe)
            ang_gt = self.GT[dic.split('.')[0]]['rot']
            x_gt, y_gt = self.GT[dic.split('.')[0]]['trans'][0], self.GT[dic.split('.')[0]]['trans'][1]
            x1, y1 = x_gt - 40 * np.sin(ang_gt), y_gt - 40 * np.cos(ang_gt)
            draw_floorplan_with_keyframe.ellipse((x_gt - 20, y_gt - 20, x_gt + 20, y_gt + 20), fill=(255, 0, 255))
            draw_floorplan_with_keyframe.line([(x_gt, y_gt), (x1, y1)], fill=(255, 0, 255), width=10)

        if self.retrieval:
            l+=70*self.plot_scale
            x_, y_ = 50*self.plot_scale, l
            x1, y1 = x_ - 20*self.plot_scale * np.sin(ang), y_ - 20 *self.plot_scale* np.cos(ang)
            draw_floorplan_with_keyframe.ellipse((x_ - 10*self.plot_scale, y_ - 10*self.plot_scale, x_ + 10*self.plot_scale, y_ + 10*self.plot_scale), fill=(255, 0, 0))
            draw_floorplan_with_keyframe.line([(x_, y_), (x1, y1)], fill=(255, 0, 0), width=int(7*self.plot_scale))
            floorplan_with_keyframe_np = np.array(floorplan_with_keyframe)
            floorplan_with_keyframe_np = cv2.putText(floorplan_with_keyframe_np, 'Similar images', (int(100*self.plot_scale), int(l)), cv2.FONT_HERSHEY_SIMPLEX,
                             1, (0, 0, 0), round(2*self.plot_scale), cv2.LINE_AA)
            floorplan_with_keyframe = Image.fromarray(floorplan_with_keyframe_np)
            draw_floorplan_with_keyframe = ImageDraw.Draw(floorplan_with_keyframe)
        if len(self.destination)>0:
            l+=70*self.plot_scale
            vertices = self.__star_vertices([50*self.plot_scale, l], 30)
            draw_floorplan_with_keyframe.polygon(vertices, fill='red')
            floorplan_with_keyframe_np = np.array(floorplan_with_keyframe)
            floorplan_with_keyframe_np = cv2.putText(floorplan_with_keyframe_np, 'Destination', (int(100*self.plot_scale), int(l)), cv2.FONT_HERSHEY_SIMPLEX,
                             1, (0, 0, 0), round(2*self.plot_scale), cv2.LINE_AA)
            
            floorplan_with_keyframe = Image.fromarray(floorplan_with_keyframe_np)
            draw_floorplan_with_keyframe = ImageDraw.Draw(floorplan_with_keyframe)
        draw_floorplan_with_keyframe.rectangle([(10,5),(400+100*self.plot_scale,l+40*self.plot_scale)],outline='black',width=int(2*self.plot_scale))     

        if self.pose:
            x,y,ang=self.pose
            ang_pi=ang/180*np.pi
            x1, y1 = x - 40*self.plot_scale * np.sin(ang_pi), y - 40*self.plot_scale * np.cos(ang_pi)
            draw_floorplan_with_keyframe.ellipse((x - 20*self.plot_scale, y - 20*self.plot_scale, x + 20*self.plot_scale, y + 20*self.plot_scale), fill=(50, 0, 106))
            draw_floorplan_with_keyframe.line([(x, y), (x1, y1)], fill=(50, 0, 106), width=int(10*self.plot_scale))
            message='Current location:  [%d,%d],  orientation:  %d degree' % (
                x, y, ang)
        else:
            message= 'Cannot localize'
        
        floorplan_with_keyframe_np = np.array(floorplan_with_keyframe) 
            
        floorplan_with_keyframe_np = cv2.putText(floorplan_with_keyframe_np, message, (10, h - 200), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 2, cv2.LINE_AA)
        if self.pose:
            floorplan_with_keyframe_np = cv2.putText(floorplan_with_keyframe_np, self.instruction_message, (10, h - 80), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (255, 0, 0), 2, cv2.LINE_AA)
                        

        if len(self.destination)>0:
            for ind,d in enumerate(self.destination):
                x_,y_=self.destination_list_location[self.destination_list_name.index(d)]
                floorplan_with_keyframe_np=cv2.putText(floorplan_with_keyframe_np, 'Destination location %d:  [%d,%d]' % (ind+1,
                            x_,y_), (10, h - 140-(len(self.destination)-ind-1)*60), cv2.FONT_HERSHEY_SIMPLEX,
                                            1, (0, 0, 0), 2, cv2.LINE_AA)

        floorplan_with_keyframe = Image.fromarray(floorplan_with_keyframe_np)
        draw_floorplan_with_keyframe = ImageDraw.Draw(floorplan_with_keyframe)

        if self.pose and len(self.destination)>0:
            paths=[self.pose[:2]]+self.paths
            for i in range(1,len(paths)):
                x0, y0=paths[i-1]
                x1, y1=paths[i]
                draw_floorplan_with_keyframe.line([(x0, y0), (x1, y1)], fill=(255,0,0), width=int(10*self.plot_scale))
                distance=np.linalg.norm([x1-x0,y1-y0])
                rot=np.arctan2(x1-x0,y1-y0)
                rot_ang=(rot-ang)/np.pi*180

        if self.retrieval:
            for i in self.hloc.retrived_image_index:
                x_,y_,ang=self.database_loc[i]
                x1, y1 = x_ - 20*self.plot_scale * np.sin(ang), y_ - 20*self.plot_scale * np.cos(ang)
                draw_floorplan_with_keyframe.ellipse((x_ - 10*self.plot_scale, y_ - 10*self.plot_scale, x_ + 10*self.plot_scale, y_ + 10*self.plot_scale), fill=(255, 0, 0))
                draw_floorplan_with_keyframe.line([(x_, y_), (x1, y1)], fill=(255, 0, 0), width=int(7*self.plot_scale))
        del self.hloc.retrived_image_index
        
        width, height = floorplan_with_keyframe.size
        scale = 1600 / width
        newsize = (1600, int(height * scale))
        floorplan_with_keyframe = floorplan_with_keyframe.resize(newsize)
        tkimage1 = ImageTk.PhotoImage(floorplan_with_keyframe)
        self.myvar1 = Label(self, image=tkimage1)
        self.myvar1.image = tkimage1
        self.myvar1.grid(row=0, column=4, columnspan=1, rowspan=40, sticky="snew")
        self.myvar1.bind('<Double-Button-1>', lambda event, action='double':
        self.select_destination(action))

    def set_destination(self):
        floorplan=self.floorplan.copy()
        draw_floorplan = ImageDraw.Draw(floorplan)
        for i,keyframe_name in enumerate(self.keyframe_name):
            if keyframe_name not in self.destination:
                x,y=self.keyframe_location[i]
                draw_floorplan.ellipse((x - 2*self.plot_scale, y - 2*self.plot_scale, x + 2*self.plot_scale, y + 2*self.plot_scale), fill=(0, 255, 0))

        for d in self.destination:
            x, y =self.keyframe_location[self.keyframe_name.index(d)]
            vertices = self.__star_vertices([x, y],30)
            draw_floorplan.polygon(vertices, fill='red')

        self.floorplan_with_keyframe = floorplan.copy()

        width, height = floorplan.size
        scale = 1600 / width
        newsize = (1600, int(height * scale))
        floorplan = floorplan.resize(newsize)
        tkimage1 = ImageTk.PhotoImage(floorplan)
        self.myvar1 = Label(self, image=tkimage1)
        self.myvar1.image = tkimage1
        self.myvar1.grid(row=0, column=4, columnspan=1, rowspan=40, sticky="snew")
        self.myvar1.bind('<Double-Button-1>', lambda event, action='double':
        self.select_destination(action))

    def animate(self,action):
        """
        Show testing image in GUI
        """
        self.value = self.lb.get(self.lb.curselection())
        if action == 'up':
            i = self.testing_image_list.index(self.value)
            if i > 0:
                self.value = self.testing_image_list[i - 1]
        if action == 'down':
            i = self.testing_image_list.index(self.value)
            if i < (len(self.testing_image_list) - 1):
                self.value = self.testing_image_list[i + 1]
        for type in self.image_types:
            q_path=join(self.testing_image_folder, self.value + type)
            if exists(q_path):
                testing_image = Image.open(q_path)
                break
        
        width, height = testing_image.size
        scale = 210 / width
        newsize = (210, int(height * scale))

        timg=testing_image.copy()
        timg = timg.resize(newsize)
        tkimage1 = ImageTk.PhotoImage(timg)
        self.myvar1 = Label(self, image=tkimage1)
        self.myvar1.image = tkimage1
        self.myvar1.grid(row=1, column=0, columnspan=1, padx=10, rowspan=5, sticky="snew")

        """
        Localize testing image
        """
        scale = 640 / width
        newsize = (640, int(height * scale))
        testing_image = testing_image.resize(newsize)
        testing_image=np.array(testing_image)
        testing_image=cv2.cvtColor(testing_image,cv2.COLOR_BGR2RGB)

        self.pose = self.hloc.get_location(testing_image) # Localize image

        """
        Navigation
        """
        if self.pose and len(self.destination)>0:
            self.paths=[]
            for destination_id in self.destination:
                path_list=self.trajectory.calculate_path(self.pose[:2], destination_id)
                if len(path_list)>0:
                    self.paths+=path_list
            
            action_list=actions(self.pose,self.paths,self.map_scale)
            
            if len(action_list)==0:
                self.instruction_message="There is no path to the destination. "
            else:
                length = action_list[0][1]
                if not self.initial:
                    self.instruction_message=command_debug(action_list)
                    self.initial = True
                    self.base_len=length
                else:
                    try:
                        self.instruction_message,self.halfway=command_count(self,action_list,length)
                    except:
                        print(action_list)

                
            x, y, an = self.pose
            print("x="+str(x)+", y="+str(y)+", angle="+str(an))
            self.logger.info(f"===============================================\n                                                       {self.instruction_message}\n                                                       ===============================================")
        elif not self.pose:
            self.instruction_message = 'Cannot localize at this point.'
            self.logger.info(f"===============================================\n                                                       {self.instruction_message}\n                                                       ===============================================")


        """
        Animate results on the floor plan
        """
        floorplan_with_keyframe=self.floorplan_with_keyframe.copy()
        self.__visualize_result(floorplan_with_keyframe)

    def select_destination(self,w):
        self.newWindow = tk.Toplevel(self.master)
        self.app1 = Destination_window(self.newWindow, parent=self)

def main(root,hloc_config,visual_config):
    map_data=loader.load_data(visual_config)
    hloc = Hloc(root, map_data, hloc_config)
    trajectory=Trajectory(map_data)

    style = Style(theme='darkly')
    root = style.master
    Main_window(root, map_data=map_data,hloc=hloc,trajectory=trajectory,**visual_config)
    root.mainloop()

if __name__=='__main__':
    root = dirname(realpath(__file__)).replace('/src','')
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--visual_config', type=str, default='configs/visualization.yaml')
    parser.add_argument('-l', '--hloc_config', type=str, default='configs/hloc.yaml')
    args = parser.parse_args()
    with open(args.hloc_config, 'r') as f:
        hloc_config = yaml.safe_load(f)
    with open(args.visual_config, 'r') as f:
        visual_config = yaml.safe_load(f)
    main(root,hloc_config,visual_config)