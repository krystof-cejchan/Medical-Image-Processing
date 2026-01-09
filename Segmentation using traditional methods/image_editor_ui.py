from functools import partial
import os
from tkinter import (
    Button, DoubleVar, IntVar, OptionMenu, StringVar, Tk, Frame, Label, Scale, HORIZONTAL, Toplevel, filedialog,
    BOTH, LEFT, RIGHT, TOP, BOTTOM, X, SUNKEN, messagebox
)
import tkinter as tk

from typing import List
from PIL import ImageTk
import numpy as np
import cv2 as cv
from image_operations import pipelines
from image_operations.advanced_operations import *
from image_operations.monadic_operations import *
from image_operations.filters import BF, bilateral_filter, gaussian_blur, mean_blur, median_filter
from image_operations.pipelines import *
from image_operations.pipelines import roi_binarization1
from image_operations.pipelines import pipeline1
from utils.histograms import show_histograms_and_cdfs
from utils.convertors import from_uint8, pil_from_float
import image_operations.pipelines



class ImageEditorApp:
    def __init__(self, root: Tk, initial_img_path: str | None = None):
        self.root = root
        self.root.title("NI-PV")

        # data
        self.original: np.ndarray | None = None
        self.edited: np.ndarray | None = None
        self.last_rois: List[np.ndarray] = []
        self.last_rois_binary: List[np.ndarray] = []

        # build menus
        menubar = self._build_menu()
        self.root.config(menu=menubar)

        # image views
        views = Frame(root)
        views.pack(side=TOP, fill=BOTH, expand=True)

        self.lbl_left = Label(views, bd=1, relief="solid")
        self.lbl_left.pack(side=LEFT, padx=8, pady=8, expand=True)
        self.lbl_right = Label(views, bd=1, relief="solid")
        self.lbl_right.pack(side=RIGHT, padx=8, pady=8, expand=True)

        # status bar
        self.status = Label(root, text="Ready", bd=1, relief=SUNKEN, anchor="w")
        self.status.pack(side=BOTTOM, fill=X)

        if initial_img_path and os.path.exists(initial_img_path):
            img_u8 = cv.imread(initial_img_path, cv.IMREAD_GRAYSCALE)
            if img_u8 is not None:
                self.original = from_uint8(img_u8)
                self.edited = self.original.copy()
        self.refresh_images()

        self._bind_shortcuts()

        self._edit_backup: np.ndarray | None = None


    def _build_menu(self):
        menubar = tk.Menu(self.root)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open…    Ctrl+O", command=self.open_image)
        file_menu.add_command(label="Save Edited Image…    Ctrl+S", command=self.save_edited_image)
        file_menu.add_command(label="Save Rectangles…    Ctrl+Shift+S", command=self.save_rects_current)
        file_menu.add_separator()
        file_menu.add_command(label="Exit    Ctrl+Q", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)

        adjust_menu = tk.Menu(menubar, tearoff=0)
        adjust_menu.add_command(label="Gamma…    Ctrl+G", command=self.open_gamma_dialog)
        adjust_menu.add_command(label="Brightness…    Ctrl+B", command=self.open_brightness_dialog)
        adjust_menu.add_command(label="Contrast…    Ctrl+Shift+C", command=self.open_contrast_dialog)
        adjust_menu.add_command(label="Non-linear Contrast...", command=self.open_non_linear_contrast_dialog)
        adjust_menu.add_command(label="Logarithmic Scale...", command=self.open_log_scale_dialog)
        adjust_menu.add_command(label="Quantization...", command=self.open_quantization_dialog)
        adjust_menu.add_command(label="OTSU", command=self.apply_otsu)
        adjust_menu.add_separator()
        adjust_menu.add_command(label="Invert (Negate)", command=self.apply_negate)
        menubar.add_cascade(label="Adjust", menu=adjust_menu)

        adv_menu = tk.Menu(menubar, tearoff=0)
        adv_menu.add_command(label="Erosion", command=self.open_erosion_dialog)
        adv_menu.add_command(label="Dilatation", command=self.open_dilatation_dialog)
        adv_menu.add_command(label="Opening", command=self.open_opening_dialog)        
        adv_menu.add_command(label="Closing", command=self.open_closing_dialog)
        adv_menu.add_command(label="Image Reconstruction", command=self.apply_img_reconstruction)
        adv_menu.add_command(label="Contours", command=self.open_contours_dialog)
        adv_menu.add_command(label="Extract Rectangles    F6", command=self.extract_rects_current)
        menubar.add_cascade(label="Process", menu=adv_menu)
        
        blurs_filters_menu = tk.Menu(menubar, tearoff=0)
        blurs_filters_menu.add_command(label="Mean blur", command=self.open_mean_blur_dialog)
        blurs_filters_menu.add_command(label="Gaussian blur", command=self.open_gaussian_blur_dialog)
        blurs_filters_menu.add_separator()
        blurs_filters_menu.add_command(label="Median filter", command=self.open_median_filter_dialog)
        blurs_filters_menu.add_command(label="Bilateral filter", command=self.open_bilateral_filter_dialog)
        blurs_filters_menu.add_separator()
        blurs_filters_menu.add_command(label="Canny Edge Detection", command=self.open_canny_dialog)
        blurs_filters_menu.add_separator()
        blurs_filters_menu.add_command(label="Difference of Gaussians", command=self.open_dog_dialog)
        blurs_filters_menu.add_command(label="Laplacian of Gaussian", command=self.open_log_dialog)

        menubar.add_cascade(label="Filters&Blurs", menu=blurs_filters_menu)
        
        hist_menu = tk.Menu(menubar, tearoff=0)
        hist_menu.add_command(label="Show histograms and CDFs", command=self.show_histograms)
        hist_menu.add_command(label="Equalize histogram", command=self.apply_hist_equalization)
        hist_menu.add_command(label="CLAHE", command=self.apply_clahe)
        
        menubar.add_cascade(label="Histograms", menu=hist_menu)

        view_menu = tk.Menu(menubar, tearoff=0)
        view_menu.add_command(label="Reset View    Ctrl+R", command=self.reset_view)
        menubar.add_cascade(label="View", menu=view_menu)
        
        pipelines = tk.Menu(menubar, tearoff=0)
        pipelines.add_command(label="Pipeline 1 - nejlepší", command=partial(self.apply_pipeline, 1, 1613, 1, 33))
        pipelines.add_command(label="Pipeline 2", command=partial(self.apply_pipeline, 2, 7000, 2))

        menubar.add_cascade(label="Pipelines", menu=pipelines)

        return menubar

    def _bind_shortcuts(self):
        self.root.bind_all("<Control-o>",       lambda e: self.open_image())
        self.root.bind_all("<Control-s>",       lambda e: self.save_edited_image())
        self.root.bind_all("<Control-Shift-s>", lambda e: self.save_rects_current())
        self.root.bind_all("<Control-q>",       lambda e: self.root.quit())
        self.root.bind_all("<Control-r>",       lambda e: self.reset_view())

    def set_status(self, text: str):
        self.status.config(text=text)

    def refresh_images(self):
        def set_img(lbl: Label, img_f: np.ndarray | None, empty_text="(no image)"):
            if img_f is None:
                lbl.config(image="", text=empty_text)
                lbl.image = None
                return
            pil = pil_from_float(img_f)
            imgtk = ImageTk.PhotoImage(pil)
            lbl.config(image=imgtk, text="")
            lbl.image = imgtk

        set_img(self.lbl_left, self.original, "(open an image)")
        set_img(self.lbl_right, self.edited, "(preview)")

    def reset_view(self):
        if self.original is None:
            self.set_status("No image loaded.")
            return
        self.edited = self.original.copy()
        self.last_rois = []
        self.refresh_images()
        self.set_status("View reset.")
        
    def show_histograms(self):
        show_histograms_and_cdfs(self.original, self.edited)


    def open_image(self):
        file_path = filedialog.askopenfilename(
            title="Select image (grayscale)",
            filetypes=[("All files", "*.*")],
        )
        if not file_path:
            self.set_status("Open canceled.")
            return
        img_u8 = cv.imread(file_path, cv.IMREAD_GRAYSCALE)
        if img_u8 is None:
            messagebox.showerror("Error", "Failed to load the file.")
            self.set_status("Failed to load image.")
            return
        self.original = from_uint8(img_u8)
        self.edited = self.original.copy()
        self.last_rois = []
        self.refresh_images()
        self.set_status(f"Loaded: {os.path.basename(file_path)}")

    def save_edited_image(self):
        if self.edited is None:
            messagebox.showwarning("Save", "Nothing to save.")
            return
        # Save as 
        path = filedialog.asksaveasfilename(
            title="Save edited image",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("TIFF", "*.tif;*.tiff"), ("JPEG", "*.jpg;*.jpeg"), ("All files", "*.*")],
        )
        if not path:
            self.set_status("Save canceled.")
            return
        ok = cv.imwrite(path, np.clip(self.edited * 255.0, 0, 255).astype(np.uint8))
        if ok:
            messagebox.showinfo("Saved", f"Saved image to:\n{path}")
            self.set_status(f"Saved: {os.path.basename(path)}")
        else:
            messagebox.showerror("Error", "Failed to save image.")
            self.set_status("Save failed.")


    def find_contours(self, img, min_area = 300):
        if self.edited is None or img is None:
            messagebox.showwarning("Contours", "Open an image first.")
            self.set_status("Open an image first.")
            return
        img = img if img is not None else self.edited
        
        _, contours = find_contours(img, min_area=min_area)
        self.set_status(f"Contours found: {len(contours)}")
        return contours

    def extract_rects_current(self, min_area = 300, padding:int=0):
        if self.edited is None:
            messagebox.showwarning("Rectangles", "Open an image first.")
            self.set_status("Open an image first.")
            return

        bin_f, contours = find_contours(self.edited, min_area=min_area)
        #self.edited = bin_f
        self.last_rois = contours_to_rect_images(self.original, contours, pad=padding)
        self.refresh_images()
        self.set_status(f"Rectangles extracted: {len(self.last_rois)}")

    def save_rects_current(self):
        if self.last_rois == None:
            messagebox.showwarning("Save", "Use Process → Extract Rectangles first.")
            self.set_status("Nothing to save")
            return
        out_dir = filedialog.askdirectory(title="Select output folder for ROIs")
        if not out_dir:
            self.set_status("Save canceled.")
            return
        count = save_rect_images(self.last_rois, out_dir, prefix="original_roi")
        _ = save_rect_images(self.last_rois_binary, out_dir, prefix="binary_roi")

        messagebox.showinfo("Saved", f"Saved {count} images to:\n{out_dir}")
        self.set_status(f"Saved {count} ROIs → {out_dir}")


    def _open_realtime_dialog(self, title: str, *,
                              scale_from: int, scale_to: int, scale_init: int,
                              value_to_param, value_to_text, op_func):
        if self.edited is None:
            messagebox.showwarning(title, "Open an image first.")
            return

        # Backup
        self._edit_backup = self.edited.copy()
        src = self._edit_backup

        dlg = Toplevel(self.root)
        dlg.title(title)
        dlg.transient(self.root)
        dlg.grab_set() 

        row = Frame(dlg)
        row.pack(side=TOP, fill=X, padx=12, pady=10)

        Label(row, text=title, width=18, anchor="w").pack(side=LEFT)
        value_label = Label(row, text=value_to_text(scale_init), width=8, anchor="e")
        value_label.pack(side=RIGHT)

        scale = Scale(dlg, from_=scale_from, to=scale_to, orient=HORIZONTAL, length=360)
        scale.set(scale_init)
        scale.pack(side=TOP, padx=12, pady=(0, 8))

        btns = Frame(dlg)
        btns.pack(side=TOP, fill=X, padx=12, pady=8)
        btn_apply = Button(btns, text="Apply", width=10)
        btn_reset = Button(btns, text="Reset", width=10)
        btn_cancel = Button(btns, text="Cancel", width=10)
        btn_apply.pack(side=RIGHT, padx=4)
        btn_cancel.pack(side=RIGHT, padx=4)
        btn_reset.pack(side=LEFT, padx=4)

        def update_preview(_=None):
            v = scale.get()
            value_label.config(text=value_to_text(v))
            param = value_to_param(v)
            preview = op_func(src, param)
            self.edited = preview
            self.refresh_images()
            self.set_status(f"{title}: {value_to_text(v)}")

        scale.config(command=lambda _: update_preview())

        def on_apply():
            dlg.destroy()
            self._edit_backup = None
            self.set_status(f"{title} applied.")

        def on_reset():
            scale.set(scale_init)

        def on_cancel():
            if self._edit_backup is not None:
                self.edited = self._edit_backup
                self.refresh_images()
            dlg.destroy()
            self._edit_backup = None
            self.set_status(f"{title} canceled.")

        btn_apply.config(command=on_apply)
        btn_reset.config(command=on_reset)
        btn_cancel.config(command=on_cancel)
        dlg.protocol("WM_DELETE_WINDOW", on_cancel)

        update_preview()

        dlg.update_idletasks()
        rx = self.root.winfo_rootx()
        ry = self.root.winfo_rooty()
        rw = self.root.winfo_width()
        rh = self.root.winfo_height()
        dw = dlg.winfo_width()
        dh = dlg.winfo_height()
        dlg.geometry(f"+{rx + (rw - dw)//2}+{ry + (rh - dh)//2}")

        dlg.wait_window(dlg)
        
    def open_contours_dialog(self):
        if self.original is None:
            messagebox.showwarning("Contours", "Open an image first.")
            return

        src = self.edited.copy()
        backup = self.edited.copy() if self.edited is not None else None

        # Dialog UI
        dlg = Toplevel(self.root)
        dlg.title("Contours")
        dlg.transient(self.root)
        dlg.grab_set()

        min_area_val = IntVar(value=0)
        pad_val = IntVar(value=0)

        row1 = Frame(dlg); row1.pack(side=TOP, fill=X, padx=12, pady=(12,6))
        Label(row1, text="Threshold (0 = Otsu)", width=20, anchor="w").pack(side=LEFT)

        row2 = Frame(dlg); row2.pack(side=TOP, fill=X, padx=12, pady=6)
        Label(row2, text="Min area", width=20, anchor="w").pack(side=LEFT)
        min_area_scale = Scale(row2, from_=0, to=50000, orient=HORIZONTAL, length=280,
                            variable=min_area_val); min_area_scale.pack(side=LEFT, padx=6)

        row3 = Frame(dlg); row3.pack(side=TOP, fill=X, padx=12, pady=6)
        Label(row3, text="Padding (px)", width=20, anchor="w").pack(side=LEFT)
        pad_scale = Scale(row3, from_=0, to=50, orient=HORIZONTAL, length=280,
                        variable=pad_val); pad_scale.pack(side=LEFT, padx=6)

        row4 = Frame(dlg); row4.pack(side=TOP, fill=X, padx=12, pady=6)
        Label(row4, text="Preview").pack(side=LEFT)

        btns = Frame(dlg); btns.pack(side=TOP, fill=X, padx=12, pady=(8,12))
        btn_apply = Button(btns, text="Apply", width=10)
        btn_reset = Button(btns, text="Reset", width=10)
        btn_cancel = Button(btns, text="Cancel", width=10)
        btn_apply.pack(side=RIGHT, padx=4)
        btn_cancel.pack(side=RIGHT, padx=4)
        btn_reset.pack(side=LEFT, padx=4)

        # Live update
        def recompute_preview(*_):
            a = max(0, min_area_val.get())
            pad = max(0, pad_val.get())

            bin_f, contours = find_contours(src, min_area=a)

            self._contours_tmp = contours
            self._binary_tmp = bin_f
            self._rois_tmp = contours_to_rect_images(src, contours, pad=pad)

            overlay = self._draw_rects_overlay(src, contours, pad=pad)
            self.edited = overlay

            self.refresh_images()
            self.set_status(f"Contours: {len(contours)} (min area {a}, pad {pad})")

        min_area_scale.config(command=lambda *_: recompute_preview())
        pad_scale.config(command=lambda *_: recompute_preview())

        def on_apply():
            self.last_rois = getattr(self, "_rois_tmp", [])
            self._contours_last = getattr(self, "_contours_tmp", [])
            dlg.destroy()
            self.set_status(f"Applied: {len(self.last_rois)} rectangles ready.")

        def on_reset():
            min_area_val.set(int(self.contour_minarea.get()))
            pad_val.set(0)
            recompute_preview()

        def on_cancel():
            if backup is not None:
                self.edited = backup
                self.refresh_images()
            dlg.destroy()
            self.set_status("Contours dialog canceled.")

        btn_apply.config(command=on_apply)
        btn_reset.config(command=on_reset)
        btn_cancel.config(command=on_cancel)
        dlg.protocol("WM_DELETE_WINDOW", on_cancel)

        recompute_preview()

        dlg.update_idletasks()
        rx, ry = self.root.winfo_rootx(), self.root.winfo_rooty()
        rw, rh = self.root.winfo_width(), self.root.winfo_height()
        dw, dh = dlg.winfo_width(), dlg.winfo_height()
        dlg.geometry(f"+{rx + (rw - dw)//2}+{ry + (rh - dh)//2}")

        dlg.wait_window(dlg)


    def _open_morph_dialog(self, title: str, *,
                        size_from: int = 1, size_to: int = 25, size_init: int = 3,
                        iter_from: int = 1, iter_to: int = 10, iter_init: int = 1,
                        op_func=None):
        if self.edited is None:
            messagebox.showwarning(title, "Open an image first.")
            return

        if op_func is None:
            messagebox.showerror(title, "Missing op_func for morphology.")
            return

        self._edit_backup = self.edited.copy()
        src = self._edit_backup

        shape_map = {
            "Rectangle": cv.MORPH_RECT,
            "Ellipse":   cv.MORPH_ELLIPSE,
            "Cross":     cv.MORPH_CROSS,
        }
        shape_names = list(shape_map.keys())

        dlg = Toplevel(self.root)
        dlg.title(title)
        dlg.transient(self.root)
        dlg.grab_set() 

        head = Frame(dlg); head.pack(side=TOP, fill=X, padx=12, pady=(10, 0))
        Label(head, text=title, width=18, anchor="w").pack(side=LEFT)
        info_label = Label(head, text="", width=24, anchor="e")
        info_label.pack(side=RIGHT)

        row_size = Frame(dlg); row_size.pack(side=TOP, fill=X, padx=12, pady=6)
        Label(row_size, text="Element size").pack(side=LEFT)
        scale_size = Scale(row_size, from_=size_from, to=size_to, orient=HORIZONTAL, length=360)
        scale_size.set(size_init)
        scale_size.pack(side=RIGHT)

        row_it = Frame(dlg); row_it.pack(side=TOP, fill=X, padx=12, pady=6)
        Label(row_it, text="Iterations").pack(side=LEFT)
        scale_iter = Scale(row_it, from_=iter_from, to=iter_to, orient=HORIZONTAL, length=360)
        scale_iter.set(iter_init)
        scale_iter.pack(side=RIGHT)

        row_shape = Frame(dlg); row_shape.pack(side=TOP, fill=X, padx=12, pady=6)
        Label(row_shape, text="Shape").pack(side=LEFT)
        shape_var = StringVar(dlg)
        shape_var.set(shape_names[0])
        shape_menu = OptionMenu(row_shape, shape_var, *shape_names)
        shape_menu.pack(side=RIGHT)

        btns = Frame(dlg); btns.pack(side=TOP, fill=X, padx=12, pady=10)
        btn_apply = Button(btns, text="Apply",  width=10)
        btn_reset = Button(btns, text="Reset",  width=10)
        btn_cancel= Button(btns, text="Cancel", width=10)
        btn_apply.pack(side=RIGHT, padx=4)
        btn_cancel.pack(side=RIGHT, padx=4)
        btn_reset.pack(side=LEFT, padx=4)

        def params_text(sz, sh_name, it):
            return f"{sh_name}, size={sz}, iter={it}"

        def update_preview(*_):
            sz = int(scale_size.get())
            it = int(scale_iter.get())
            sh_name = shape_var.get()
            sh = shape_map[sh_name]

            preview = op_func(src, sz, sh, it)
            self.edited = preview
            self.refresh_images()
            info_label.config(text=params_text(sz, sh_name, it))
            self.set_status(f"{title}: {params_text(sz, sh_name, it)}")

        scale_size.config(command=lambda _: update_preview())
        scale_iter.config(command=lambda _: update_preview())
        shape_var.trace_add("write", update_preview)

        def on_apply():
            dlg.destroy()
            self._edit_backup = None
            self.set_status(f"{title} applied.")

        def on_reset():
            scale_size.set(size_init)
            scale_iter.set(iter_init)
            shape_var.set(shape_names[0])

        def on_cancel():
            if self._edit_backup is not None:
                self.edited = self._edit_backup
                self.refresh_images()
            dlg.destroy()
            self._edit_backup = None
            self.set_status(f"{title} canceled.")

        btn_apply.config(command=on_apply)
        btn_reset.config(command=on_reset)
        btn_cancel.config(command=on_cancel)
        dlg.protocol("WM_DELETE_WINDOW", on_cancel)

        update_preview()

        dlg.update_idletasks()
        rx = self.root.winfo_rootx(); ry = self.root.winfo_rooty()
        rw = self.root.winfo_width(); rh = self.root.winfo_height()
        dw = dlg.winfo_width();       dh = dlg.winfo_height()
        dlg.geometry(f"+{rx + (rw - dw)//2}+{ry + (rh - dh)//2}")
        dlg.wait_window(dlg)
        
    def _open_clahe_dialog(self, title: str, *,
                        clip_limit_from: int = 1, clip_limit_to: int = 25, clip_limit_init: int = 3,
                        tile_grid_from: int = 1, tile_grid_to: int = 10, tile_grid_init: int = 1,
                        op_func=None):
        if self.edited is None:
            messagebox.showwarning(title, "Open an image first.")
            return

        if op_func is None:
            messagebox.showerror(title, "Missing op_func.")
            return

        self._edit_backup = self.edited.copy()
        src = self._edit_backup


        dlg = Toplevel(self.root)
        dlg.title(title)
        dlg.transient(self.root)
        dlg.grab_set()   

        head = Frame(dlg); head.pack(side=TOP, fill=X, padx=12, pady=(10, 0))
        Label(head, text=title, width=18, anchor="w").pack(side=LEFT)
        info_label = Label(head, text="", width=24, anchor="e")
        info_label.pack(side=RIGHT)

        row_size = Frame(dlg); row_size.pack(side=TOP, fill=X, padx=12, pady=6)
        Label(row_size, text="Clip limit").pack(side=LEFT)
        scale_size = Scale(row_size, from_=clip_limit_from, to=clip_limit_to, orient=HORIZONTAL, length=360)
        scale_size.set(clip_limit_init)
        scale_size.pack(side=RIGHT)

        row_it = Frame(dlg); row_it.pack(side=TOP, fill=X, padx=12, pady=6)
        Label(row_it, text="Tile Grid").pack(side=LEFT)
        scale_iter = Scale(row_it, from_=tile_grid_from, to=tile_grid_to, orient=HORIZONTAL, length=360)
        scale_iter.set(tile_grid_init)
        scale_iter.pack(side=RIGHT)

        btns = Frame(dlg); btns.pack(side=TOP, fill=X, padx=12, pady=10)
        btn_apply = Button(btns, text="Apply",  width=10)
        btn_reset = Button(btns, text="Reset",  width=10)
        btn_cancel= Button(btns, text="Cancel", width=10)
        btn_apply.pack(side=RIGHT, padx=4)
        btn_cancel.pack(side=RIGHT, padx=4)
        btn_reset.pack(side=LEFT, padx=4)

        def params_text(cl, tg):
            return f"clip limit={cl}, tile grid={tg}"

        def update_preview(*_):
            sz = int(scale_size.get())
            it = int(scale_iter.get())

            preview = op_func(src, sz, it)
            self.edited = preview
            self.refresh_images()
            info_label.config(text=params_text(sz, it))
            self.set_status(f"{title}: {params_text(sz, it)}")

        scale_size.config(command=lambda _: update_preview())
        scale_iter.config(command=lambda _: update_preview())

        def on_apply():
            dlg.destroy()
            self._edit_backup = None
            self.set_status(f"{title} applied.")

        def on_reset():
            scale_size.set(clip_limit_init)
            scale_iter.set(tile_grid_init)

        def on_cancel():
            if self._edit_backup is not None:
                self.edited = self._edit_backup
                self.refresh_images()
            dlg.destroy()
            self._edit_backup = None
            self.set_status(f"{title} canceled.")

        btn_apply.config(command=on_apply)
        btn_reset.config(command=on_reset)
        btn_cancel.config(command=on_cancel)
        dlg.protocol("WM_DELETE_WINDOW", on_cancel)

        update_preview()

        dlg.update_idletasks()
        rx = self.root.winfo_rootx(); ry = self.root.winfo_rooty()
        rw = self.root.winfo_width(); rh = self.root.winfo_height()
        dw = dlg.winfo_width();       dh = dlg.winfo_height()
        dlg.geometry(f"+{rx + (rw - dw)//2}+{ry + (rh - dh)//2}")
        dlg.wait_window(dlg)

    def _draw_rects_overlay(self, base_img_f: np.ndarray, contours, pad: int = 0) -> np.ndarray:
        img_u8 = (np.clip(base_img_f * 255.0, 0, 255)).astype(np.uint8).copy()
        h, w = img_u8.shape[:2]

        for c in contours:
            x, y, bw, bh = cv.boundingRect(c)
            x0 = max(0, x - pad)
            y0 = max(0, y - pad)
            x1 = min(w - 1, x + bw + pad)
            y1 = min(h - 1, y + bh + pad)

            cx = int(x0 + (x1 - x0) / 2)
            cy = int(y0 + (y1 - y0) / 2)

            cv.drawMarker(img_u8, (cx, cy), (0, 0, 255), cv.MARKER_CROSS, 300, 30)

        return img_u8.astype(np.float32) / 255.0


    def open_gamma_dialog(self):
        self._open_realtime_dialog(
            "Gamma",
            scale_from=10, scale_to=500, scale_init=100,
            value_to_param=lambda v: v / 100.0,
            value_to_text=lambda v: f"{v/100.0:.2f}",
            op_func=adjust_gamma
        )

    def open_brightness_dialog(self):
        self._open_realtime_dialog(
            "Brightness",
            scale_from=-100, scale_to=100, scale_init=0,
            value_to_param=lambda v: v / 100.0,
            value_to_text=lambda v: f"{v/100.0:+.2f}",
            op_func=adjust_brightness
        )

    def open_contrast_dialog(self):
        self._open_realtime_dialog(
            "Contrast",
            scale_from=10, scale_to=300, scale_init=100,    
            value_to_param=lambda v: v / 100.0,
            value_to_text=lambda v: f"{v/100.0:.2f}",
            op_func=adjust_contrast
        )
        
    def open_non_linear_contrast_dialog(self):
        self._open_realtime_dialog(
            "Non-Linear Contrast",
            scale_from=1, scale_to=99, scale_init=1,     
            value_to_param=lambda v: v / 100.0,
            value_to_text=lambda v: f"{v/100.0:.2f}",
            op_func=non_linear_contrast
        )
        
    def open_log_scale_dialog(self):
        self._open_realtime_dialog(
            "Logarithmic Scale",
            scale_from=-99, scale_to=500, scale_init=100,     
            value_to_param=lambda v: v / 100.0,
            value_to_text=lambda v: f"{v/100.0:.2f}",
            op_func=logarithmic_scale
        )
    
    def open_quantization_dialog(self):
        self._open_realtime_dialog(
            "Quantization",
            scale_from=101, scale_to=1000, scale_init=100,     
            value_to_param=lambda v: v / 100.0,
            value_to_text=lambda v: f"{v/100.0:.2f}",
            op_func=quantization
        )
    

    def open_log_dialog(self):
        self._open_realtime_dialog(
        "LoG (sigma)",
        scale_from=10, scale_to=800, scale_init=150,  
        value_to_param=lambda v: v / 100.0,
        value_to_text=lambda v: f"{v/100.0:.2f}",
        op_func=lambda img, sigma: laplacian_of_gaus(img, sigma=sigma,ksize_lap=3)
    )
    def open_dog_dialog(self):
        if self.edited is None:
            messagebox.showwarning("DoG", "Open an image first.")
            return

        src = self.edited.copy()
        backup = self.edited.copy()

        dlg = Toplevel(self.root)
        dlg.title("DoG")
        dlg.transient(self.root); dlg.grab_set()

        row1 = Frame(dlg); row1.pack(side=TOP, fill=X, padx=12, pady=(12,6))
        Label(row1, text="sigma").pack(side=LEFT)
        s_var = tk.IntVar(value=150)  
        s_scale = Scale(row1, from_=10, to=1500, orient=HORIZONTAL, length=320, variable=s_var)
        s_scale.pack(side=RIGHT)

        row2 = Frame(dlg); row2.pack(side=TOP, fill=X, padx=12, pady=6)
        Label(row2, text="k (* sigma)").pack(side=LEFT)
        k_var = tk.IntVar(value=160)  
        k_scale = Scale(row2, from_=110, to=800, orient=HORIZONTAL, length=320, variable=k_var)
        k_scale.pack(side=RIGHT)

        info = Label(dlg, text="sigma=1.50, k=1.60"); info.pack(side=TOP, padx=12)

        def _restore(self, img):
            self.edited = img
            self.refresh_images()
        btns = Frame(dlg); btns.pack(side=TOP, fill=X, padx=12, pady=10)
        Button(btns, text="Apply",  width=10, command=lambda: (dlg.destroy(), None)).pack(side=RIGHT, padx=4)
        Button(btns, text="Cancel", width=10, command=lambda: (_restore(self, backup), dlg.destroy())).pack(side=RIGHT, padx=4)
        Button(btns, text="Reset",  width=10, command=lambda: (s_var.set(150), k_var.set(160))).pack(side=LEFT, padx=4)

        def _update(*_):
            sigma = s_var.get() / 100.0
            k     = max(1.01, k_var.get() / 100.0)
            self.edited = diff_of_gaus(src, sigma=sigma, k=k)
            self.refresh_images()
            info.config(text=f"sigma={sigma:.2f}, k={k:.2f}")
            self.set_status(f"DoG: sigma={sigma:.2f}, k={k:.2f}")

        s_scale.config(command=lambda _: _update())
        k_scale.config(command=lambda _: _update())

        def _on_close():
            _restore(self, backup)
            dlg.destroy()
        dlg.protocol("WM_DELETE_WINDOW", _on_close)

        _update()




    def apply_negate(self):
        if self.edited is None:
            messagebox.showwarning("Invert", "Open an image first.")
            return
        self.edited = negate(self.edited)
        self.refresh_images()
        self.set_status("Inverted (negated).")
        
    def apply_hist_equalization(self):
        if self.edited is None:
            messagebox.showwarning("Histogram Equalization", "Open an image first.")
            return
        self.edited = hist_equalization(self.edited)
        self.refresh_images()
        self.set_status("Histogram Equalized")
        
    def apply_otsu(self):
        if self.edited is None:
            messagebox.showwarning("OTSU", "Open an image first.")
            return
        self.edited = otsu(self.edited)
        self.refresh_images()
        self.set_status("OTSU applied")
        
    def apply_clahe(self):
        self._open_clahe_dialog("CLAHE",
                                clip_limit_from=1,clip_limit_to=100, clip_limit_init=3,
                                tile_grid_from=1, tile_grid_to=25, tile_grid_init=1,
                                op_func=clahe)
    
    def apply_img_reconstruction(self):
        if self.edited is None:
            messagebox.showwarning("Image reconstruction", "Open an image first.")
            return
        self.edited = image_reconstruct(None, self.edited)
        self.refresh_images()
        self.set_status("Image reconstructed")
        

    def open_erosion_dialog(self):
        self._open_morph_dialog(
            "Erosion",
            size_from=1, size_to=500, size_init=1,     
            iter_from=1, iter_to=50, iter_init=1,
            op_func=erosion
        )
    def open_dilatation_dialog(self):
        self._open_morph_dialog(
            "Dilatation",
            size_from=1, size_to=500, size_init=1,     
            iter_from=1, iter_to=50, iter_init=1,
            op_func=dilatation
        )
    def open_opening_dialog(self):
        self._open_morph_dialog(
            "Opening",
            size_from=1, size_to=500, size_init=1,     
            iter_from=1, iter_to=50, iter_init=1,
            op_func=opening
        )

    def open_closing_dialog(self):
        self._open_morph_dialog(
            "Closing",
            size_from=1, size_to=500, size_init=1,     
            iter_from=1, iter_to=50, iter_init=1,
            op_func=closing
        )

    def open_mean_blur_dialog(self):
        if self.edited is None:
            messagebox.showwarning("Mean Blur", "Open an image first.")
            return

        src = self.edited.copy()
        backup = src.copy()

        dlg = Toplevel(self.root)
        dlg.title("Mean Blur")
        dlg.transient(self.root)
        dlg.grab_set()

        ksize_val = IntVar(value=3)

        row = Frame(dlg); row.pack(fill=X, padx=12, pady=6)
        Label(row, text="Kernel Size", width=16).pack(side=LEFT)
        scale = Scale(row, from_=1, to=25, orient=HORIZONTAL, length=300, variable=ksize_val)
        scale.pack(side=RIGHT)

        btns = Frame(dlg); btns.pack(fill=X, padx=12, pady=10)
        Button(btns, text="Apply", width=10, command=lambda: on_apply()).pack(side=RIGHT, padx=4)
        Button(btns, text="Cancel", width=10, command=lambda: on_cancel()).pack(side=RIGHT, padx=4)
        Button(btns, text="Reset", width=10, command=lambda: on_reset()).pack(side=LEFT, padx=4)

        def recompute_preview(*_):
            k = max(1, ksize_val.get() | 1) 
            preview = cv.blur(src, (k, k))
            self.edited = preview
            self.refresh_images()
            self.set_status(f"Mean Blur (ksize={k})")

        def on_apply():
            dlg.destroy()
            self.set_status(f"Applied Mean Blur (ksize={ksize_val.get()})")

        def on_cancel():
            self.edited = backup
            self.refresh_images()
            dlg.destroy()
            self.set_status("Mean Blur canceled")

        def on_reset():
            ksize_val.set(3)

        scale.config(command=recompute_preview)
        recompute_preview()
        dlg.wait_window(dlg)


    def open_gaussian_blur_dialog(self):
        if self.edited is None:
            messagebox.showwarning("Gaussian Blur", "Open an image first.")
            return

        src = self.edited.copy()
        backup = src.copy()

        dlg = Toplevel(self.root)
        dlg.title("Gaussian Blur")
        dlg.transient(self.root)
        dlg.grab_set()

        ksize_val = IntVar(value=5)
        sigma_val = DoubleVar(value=1.0)

        row1 = Frame(dlg); row1.pack(fill=X, padx=12, pady=6)
        Label(row1, text="Kernel Size", width=16).pack(side=LEFT)
        scale_k = Scale(row1, from_=1, to=31, orient=HORIZONTAL, length=300, variable=ksize_val)
        scale_k.pack(side=RIGHT)

        row2 = Frame(dlg); row2.pack(fill=X, padx=12, pady=6)
        Label(row2, text="Sigma", width=16).pack(side=LEFT)
        scale_s = Scale(row2, from_=0, to=10, resolution=0.1, orient=HORIZONTAL, length=300, variable=sigma_val)
        scale_s.pack(side=RIGHT)

        btns = Frame(dlg); btns.pack(fill=X, padx=12, pady=10)
        Button(btns, text="Apply", width=10, command=lambda: on_apply()).pack(side=RIGHT, padx=4)
        Button(btns, text="Cancel", width=10, command=lambda: on_cancel()).pack(side=RIGHT, padx=4)
        Button(btns, text="Reset", width=10, command=lambda: on_reset()).pack(side=LEFT, padx=4)

        def recompute_preview(*_):
            k = max(1, ksize_val.get() | 1)
            s = sigma_val.get()
            preview = cv.GaussianBlur(src, (k, k), sigmaX=s, sigmaY=s)
            self.edited = preview
            self.refresh_images()
            self.set_status(f"Gaussian Blur (ksize={k}, sigma={s:.2f})")

        def on_apply():
            dlg.destroy()
            self.set_status(f"Applied Gaussian Blur (ksize={ksize_val.get()}, sigma={sigma_val.get():.2f})")

        def on_cancel():
            self.edited = backup
            self.refresh_images()
            dlg.destroy()
            self.set_status("Gaussian Blur canceled")

        def on_reset():
            ksize_val.set(5)
            sigma_val.set(1.0)

        scale_k.config(command=recompute_preview)
        scale_s.config(command=recompute_preview)
        recompute_preview()
        dlg.wait_window(dlg)


    def open_median_filter_dialog(self):
        if self.edited is None:
            messagebox.showwarning("Median Filter", "Open an image first.")
            return

        src = self.edited.copy()
        backup = src.copy()

        dlg = Toplevel(self.root)
        dlg.title("Median Filter")
        dlg.transient(self.root)
        dlg.grab_set()

        ksize_val = IntVar(value=3)

        row = Frame(dlg); row.pack(fill=X, padx=12, pady=6)
        Label(row, text="Kernel Size", width=16).pack(side=LEFT)
        scale = Scale(row, from_=1, to=25, orient=HORIZONTAL, length=300, variable=ksize_val)
        scale.pack(side=RIGHT)

        btns = Frame(dlg); btns.pack(fill=X, padx=12, pady=10)
        Button(btns, text="Apply", width=10, command=lambda: on_apply()).pack(side=RIGHT, padx=4)
        Button(btns, text="Cancel", width=10, command=lambda: on_cancel()).pack(side=RIGHT, padx=4)
        Button(btns, text="Reset", width=10, command=lambda: on_reset()).pack(side=LEFT, padx=4)

        def recompute_preview(*_):
            k = max(1, ksize_val.get() | 1)
            preview = cv.medianBlur(src, k)
            self.edited = preview
            self.refresh_images()
            self.set_status(f"Median Filter (ksize={k})")

        def on_apply():
            dlg.destroy()
            self.set_status(f"Applied Median Filter (ksize={ksize_val.get()})")

        def on_cancel():
            self.edited = backup
            self.refresh_images()
            dlg.destroy()
            self.set_status("Median Filter canceled")

        def on_reset():
            ksize_val.set(3)

        scale.config(command=recompute_preview)
        recompute_preview()
        dlg.wait_window(dlg)


    def open_bilateral_filter_dialog(self):
        if self.edited is None:
            messagebox.showwarning("Bilateral Filter", "Open an image first.")
            return

        src = self.edited.copy()
        backup = src.copy()

        dlg = Toplevel(self.root)
        dlg.title("Bilateral Filter")
        dlg.transient(self.root)
        dlg.grab_set()

        d_val = IntVar(value=9)
        sigma_color_val = DoubleVar(value=75.0)
        sigma_space_val = DoubleVar(value=75.0)

        for label_text, var, frm, to, res in [
            ("Diameter", d_val, 1, 25, 1),
            ("Sigma Color", sigma_color_val, 1, 150, 1),
            ("Sigma Space", sigma_space_val, 1, 150, 1)
        ]:
            row = Frame(dlg); row.pack(fill=X, padx=12, pady=6)
            Label(row, text=label_text, width=16).pack(side=LEFT)
            Scale(row, from_=frm, to=to, resolution=res,
                orient=HORIZONTAL, length=300, variable=var,
                command=lambda *_: recompute_preview()).pack(side=RIGHT)

        btns = Frame(dlg); btns.pack(fill=X, padx=12, pady=10)
        Button(btns, text="Apply", width=10, command=lambda: on_apply()).pack(side=RIGHT, padx=4)
        Button(btns, text="Cancel", width=10, command=lambda: on_cancel()).pack(side=RIGHT, padx=4)
        Button(btns, text="Reset", width=10, command=lambda: on_reset()).pack(side=LEFT, padx=4)

        def recompute_preview(*_):
            d = d_val.get()
            sc = sigma_color_val.get()
            ss = sigma_space_val.get()
            preview = cv.bilateralFilter(src, d=d, sigmaColor=sc, sigmaSpace=ss)
            self.edited = preview
            self.refresh_images()
            self.set_status(f"Bilateral Filter (d={d}, sigmaC={sc}, sigmaS={ss})")

        def on_apply():
            dlg.destroy()
            self.set_status("Applied Bilateral Filter")

        def on_cancel():
            self.edited = backup
            self.refresh_images()
            dlg.destroy()
            self.set_status("Bilateral Filter canceled")

        def on_reset():
            d_val.set(9)
            sigma_color_val.set(75)
            sigma_space_val.set(75)
            recompute_preview()

        recompute_preview()
        dlg.wait_window(dlg)


    def open_canny_dialog(self):
        if self.edited is None:
            messagebox.showwarning("Canny", "Open an image first.")
            return

        src = self.edited.copy()
        backup = src.copy()

        dlg = Toplevel(self.root)
        dlg.title("Canny Edge Detection")
        dlg.transient(self.root)
        dlg.grab_set()

        t1_val = IntVar(value=50)
        t2_val = IntVar(value=150)

        row1 = Frame(dlg); row1.pack(fill=X, padx=12, pady=6)
        Label(row1, text="Threshold 1", width=16).pack(side=LEFT)
        scale1 = Scale(row1, from_=0, to=255, orient=HORIZONTAL, length=300, variable=t1_val)
        scale1.pack(side=RIGHT)

        row2 = Frame(dlg); row2.pack(fill=X, padx=12, pady=6)
        Label(row2, text="Threshold 2", width=16).pack(side=LEFT)
        scale2 = Scale(row2, from_=0, to=255, orient=HORIZONTAL, length=300, variable=t2_val)
        scale2.pack(side=RIGHT)

        btns = Frame(dlg); btns.pack(fill=X, padx=12, pady=10)
        Button(btns, text="Apply", width=10, command=lambda: on_apply()).pack(side=RIGHT, padx=4)
        Button(btns, text="Cancel", width=10, command=lambda: on_cancel()).pack(side=RIGHT, padx=4)
        Button(btns, text="Reset", width=10, command=lambda: on_reset()).pack(side=LEFT, padx=4)

        def recompute_preview(*_):
            t1 = t1_val.get()
            t2 = t2_val.get()
            img8 = (src * 255).astype(np.uint8) if src.max() <= 1 else src.astype(np.uint8)
            preview = cv.Canny(img8, t1, t2)
            self.edited = preview.astype(np.float32) / 255.0
            self.refresh_images()
            self.set_status(f"Canny (t1={t1}, t2={t2})")

        def on_apply():
            dlg.destroy()
            self.set_status(f"Applied Canny (t1={t1_val.get()}, t2={t2_val.get()})")

        def on_cancel():
            self.edited = backup
            self.refresh_images()
            dlg.destroy()
            self.set_status("Canny canceled")

        def on_reset():
            t1_val.set(50)
            t2_val.set(150)
            recompute_preview()

        scale1.config(command=recompute_preview)
        scale2.config(command=recompute_preview)
        recompute_preview()
        dlg.wait_window(dlg)

    def split_cells(self):
        if self.edited is None:
            messagebox.showwarning("Split cells", "Open an image first.")
            return
        self.edited = split_touching(self.edited)
        self.refresh_images()
        self.set_status("Cells splitted")

    def _apply_filter(self, bf: BF):
        if self.edited is None:
            if self.original is None:
                messagebox.showwarning("Filters&Blurs", "Open an image first.")
                return
            self.edited = self.original
        edited_copy = self.edited.copy()
        match bf:
            case BF.B_MEAN:
                edited_copy = mean_blur(edited_copy)
            case BF.B_GAUSSIAN:
                edited_copy = gaussian_blur(edited_copy)
            case BF.F_MEDIAN:
                edited_copy = median_filter(edited_copy)
            case BF.F_BILATERAL:
                edited_copy = bilateral_filter(edited_copy)
        self.edited = edited_copy
        self.refresh_images()
        self.set_status("Filters and blurs applied.")

    def apply_blur_mean(self):
        self._apply_filter(BF.B_MEAN)
    def apply_blur_gaus(self):
        self._apply_filter(BF.B_GAUSSIAN)
    def apply_filter_median(self):
        self._apply_filter(BF.F_MEDIAN)
    def apply_filter_bilateral(self):
        self._apply_filter(BF.F_BILATERAL) 
    def apply_canny(self):
        self._open_morph_dialog(
            "Canny",
            size_from=1, size_to=500, size_init=1,     
            iter_from=1, iter_to=50, iter_init=1,
            op_func=canny
        )
    
    def apply_pipeline(self, pipeline_number: int, contour_rect_size:int, binarization_method:int, contour_bin_cell_size:int=150):
        fun = getattr(pipelines, f'pipeline{pipeline_number}')# vzhledej pipeline podle jména funkce
        self.edited = fun(self.original) # spusť pipeline na vstupním obrázu; výsledek ulož do upraveného obrázku
        self.extract_rects_current(contour_rect_size,10) #pomocí obdélníkových kontour najdi "buňky"
        self.last_rois_binary.clear() 
        for i in range(len(self.last_rois) - 1, -1, -1):#pro každou najitou buňku...
            roi = self.last_rois[i]
            fun_bin = getattr(pipelines, f'roi_binarization{binarization_method}')
            binarized_roi = fun_bin(roi) #...spusť binarizaci buňky
            binarized_roi = split_middle_cell(binarized_roi) #...rozděl prostřední buňku
            contours = self.find_contours(binarized_roi, contour_bin_cell_size) #..pomocí kontour najdi "jádra buňky"
             #odstraň vše, co není v konturách
            mask = np.zeros(binarized_roi.shape[:2], dtype=np.uint8)
            cv.drawContours(mask, contours, -1, 1, thickness=cv.FILLED)
            binarized_roi[mask == 0] = 0
            if not (len(contours)==11): #pokud je buňka vadná, tj. neobsahuje 11 jader, vymaž ji
                self.last_rois.pop(i)
                continue
            counting = 0
            for c in contours:
                # očíseluj jádra a do pravého horního rohu napiš celkový počet jader
                x, y, bw, bh = cv.boundingRect(c)
                h, w = binarized_roi.shape[:2]

                x0 = max(0, x)
                y0 = max(0, y)
                x1 = min(w - 1, x + bw)
                y1 = min(h - 1, y + bh)

                cx = int(x0 + (x1 - x0) / 2)
                cy = int(y0 + (y1 - y0) / 2)
                
                counting +=1

                cv.putText(binarized_roi, str(counting), (cx, cy), cv.FONT_HERSHEY_PLAIN,0.5, 0,1)
            cv.putText(binarized_roi, str(counting), (5, 5), cv.FONT_HERSHEY_PLAIN,0.5, 1,1)
            self.last_rois_binary.append(binarized_roi)
        self.save_rects_current() #otevře dialog pro vybrání cílového adresáře a uloží tam output (binarizované a grey-scale obrázky  = dataset)
