import tkinter as tk
from PIL import Image,ImageDraw
import os

class ImageGenerator:
    def __init__(self,parent,posx,posy,*kwargs):
        self.parent = parent
        self.posx = posx
        self.posy = posy
        self.sizex = 200
        self.sizey = 200
        self.b1 = "up"
        self.xold = None
        self.yold = None 
        self.drawing_area=tk.Canvas(self.parent,width=self.sizex,height=self.sizey)
        self.drawing_area.place(x=self.posx,y=self.posy)
        self.drawing_area.bind("<Motion>", self.motion)
        self.drawing_area.bind("<ButtonPress-1>", self.b1down)
        self.drawing_area.bind("<ButtonRelease-1>", self.b1up)
        self.button=tk.Button(self.parent,text="Done!",width=10,bg='white',command=self.save)
        self.button.place(x=self.sizex/7,y=self.sizey+20)
        self.button1=tk.Button(self.parent,text="Clear!",width=10,bg='white',command=self.clear)
        self.button1.place(x=(self.sizex/7)+80,y=self.sizey+20)

        self.image=Image.new("RGB",(200,200),(255,255,255))
        self.draw=ImageDraw.Draw(self.image)

    # save image user sketched
    def save(self):
        cwd = os.getcwd()
        user_img_dir = os.path.join(cwd, "user_imgs")

        # make dir for user images if not already exists
        if not os.path.exists(user_img_dir):
            os.mkdir(user_img_dir)

        filename = "temp.png"
        self.image.save(os.path.join(user_img_dir, filename))

    # clear paint workspace
    def clear(self):
        self.drawing_area.delete("all")
        self.image=Image.new("RGB",(200,200),(255,255,255))
        self.draw=ImageDraw.Draw(self.image)

    def b1down(self,event):
        self.b1 = "down"

    def b1up(self,event):
        self.b1 = "up"
        self.xold = None
        self.yold = None

    def motion(self,event):
        if self.b1 == "down":
            if self.xold is not None and self.yold is not None:
                event.widget.create_line(self.xold,self.yold,event.x,event.y,smooth='true',width=3,fill='black')
                self.draw.line(((self.xold,self.yold),(event.x,event.y)),(0,128,0),width=3)

        self.xold = event.x
        self.yold = event.y

if __name__ == "__main__":
    root=tk.Tk()
    root.wm_geometry("%dx%d+%d+%d" % (400, 400, 10, 10))
    root.config(bg='white')
    ImageGenerator(root,10,10)

    # feed user image into model and output label
    textbox = tk.Text(root, height=2, width=30)
    textbox.pack(side=tk.RIGHT)
    textbox.insert(tk.END, "Just a text Widget\nin two lines\n")

    root.mainloop()