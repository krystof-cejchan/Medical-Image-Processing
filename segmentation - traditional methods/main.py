from tkinter import Tk
from image_editor_ui import ImageEditorApp


if __name__ == "__main__":
    root = Tk()
    app = ImageEditorApp(root, initial_img_path=None)
    root.mainloop()
