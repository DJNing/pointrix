import os
import dearpygui.dearpygui as dpg
import numpy as np
import torch
import tqdm


def register_dpg(gui):
    ### register texture
    with dpg.texture_registry(show=False):
        dpg.add_raw_texture(
            gui.W,
            gui.H,
            gui.buffer_image,
            format=dpg.mvFormat_Float_rgb,
            tag="_texture",
        )

    ### register window
    # the rendered image, as the primary window
    with dpg.window(
        tag="_primary_window",
        width=gui.W,
        height=gui.H,
        pos=[0, 0],
        no_move=True,
        no_title_bar=True,
        no_scrollbar=True,
    ):
        # add the texture
        dpg.add_image("_texture")

    # dpg.set_primary_window("_primary_window", True)

    # control window
    with dpg.window(
        label="Control",
        tag="_control_window",
        width=600,
        height=gui.H,
        pos=[gui.W, 0],
        no_move=True,
        no_title_bar=True,
    ):
        # button theme
        with dpg.theme() as theme_button:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (23, 3, 18))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (51, 3, 47))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (83, 18, 83))
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)

        # timer stuff
        with dpg.group(horizontal=True):
            dpg.add_text("Infer time: ")
            dpg.add_text("no data", tag="_log_infer_time")

        def callback_setattr(sender, app_data, user_data):
            setattr(gui, user_data, app_data)

        # init stuff
        with dpg.collapsing_header(label="Initialize", default_open=True):

            # seed stuff
            def callback_set_seed(sender, app_data):
                gui.seed = app_data
                gui.seed_everything()

            dpg.add_input_text(
                label="seed",
                default_value=gui.seed,
                on_enter=True,
                callback=callback_set_seed,
            )

            # >>>>>>>>>>>>>>>>>>>>>  input stuff
            def callback_select_input(sender, app_data):
                # only one item
                for k, v in app_data["selections"].items():
                    dpg.set_value("_log_input", k)
                    gui.load_input(v)

                gui.need_update = True

            with dpg.file_dialog(
                directory_selector=False,
                show=False,
                callback=callback_select_input,
                file_count=1,
                tag="file_dialog_tag",
                width=700,
                height=400,
            ):
                dpg.add_file_extension("Images{.jpg,.jpeg,.png}")

            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="input",
                    callback=lambda: dpg.show_item("file_dialog_tag"),
                )
                dpg.add_text("", tag="_log_input")

            # >>>>>>>>>>>>>>>>>>>>>  Visualization mode
            with dpg.group(horizontal=True):
                dpg.add_text("Visualization: ")

                def callback_vismode(sender, app_data, user_data):
                    gui.visualization_mode = user_data

                dpg.add_button(
                    label="RGB",
                    tag="_button_vis_rgb",
                    callback=callback_vismode,
                    user_data='RGB',
                )
                dpg.bind_item_theme("_button_vis_rgb", theme_button)

                dpg.add_button(
                    label="Depth",
                    tag="_button_vis_depth",
                    callback=callback_vismode,
                    user_data='Depth',
                )
                dpg.bind_item_theme("_button_vis_depth", theme_button)


            # >>>>>>>>>>>>>>>>>>>>> GS scale const
            with dpg.group(horizontal=True):
                dpg.add_text("Scale Const: ")
                def callback_vis_scale_const(sender):
                    gui.vis_scale_const = 10 ** dpg.get_value(sender)
                    gui.need_update = True
                dpg.add_slider_float(
                    label="Log vis_scale_const (For debugging)",
                    default_value=0,
                    max_value=1,
                    min_value=-2,
                    callback=callback_vis_scale_const,
                )


        # >>>>>>>>>>>>>>>>>>>>> Train stuff
        with dpg.collapsing_header(label="Train", default_open=True):
            # lr and train button
            with dpg.group(horizontal=True):
                dpg.add_text("Train: ")

                def callback_train(sender, app_data):
                    if gui.training:
                        gui.training = False
                        print(" ... Stop training ... ")
                        dpg.configure_item("_button_train", label="start")
                    else:
                        # gui.prepare_train()
                        gui.training = True
                        print(" ... Start training ... ")
                        dpg.configure_item("_button_train", label="stop")

                dpg.add_button(
                    label="start", tag="_button_train", callback=callback_train
                )
                dpg.bind_item_theme("_button_train", theme_button)

            # with dpg.group(horizontal=True):
            #     dpg.add_text("", tag="_log_train_psnr")
            # with dpg.group(horizontal=True):
            #     dpg.add_text("", tag="_log_train_log")

            ################## video player
            with dpg.group(horizontal=True):
                dpg.add_text("Video: ")

                ####### The play button
                def callback_play(sender, app_data):
                    gui.is_play = not gui.is_play
                dpg.add_button(
                    label="play", tag="_button_play", callback=callback_play
                )
                dpg.bind_item_theme("_button_play", theme_button)

                ####### The fix camera button
                def callback_fix_cam(sender, app_data):
                    gui.fix_cam = not gui.fix_cam
                dpg.add_button(
                    label="fix_cam", tag="_button_fix_cam", callback=callback_fix_cam
                )
                dpg.bind_item_theme("_button_fix_cam", theme_button)

                ####### The record button
                def callback_record(sender, app_data):
                    gui.record = True
                dpg.add_button(
                    label="record", tag="_button_record", callback=callback_record
                )
                dpg.bind_item_theme("_button_record", theme_button)


            ###### The video slider
            with dpg.group(horizontal=True):
                dpg.add_text("Temporal Speed: ")
                def callback_speed_control(sender):
                    # gui.video_speed = 10 * dpg.get_value(sender)
                    gui.current_fid_ratio = dpg.get_value(sender)
                    gui.need_update = True
                dpg.add_slider_float(
                    label="Play speed",
                    default_value=0.,
                    max_value=1.,
                    min_value=0.,
                    callback=callback_speed_control,
                )

            


            # >>>>>>>>>>>>>>>>>>>>> Save model / Screenshot
            with dpg.group(horizontal=True):
                dpg.add_text("Save: ")

                def callback_save(sender, app_data, user_data):
                    print("\n[ITER {}] Saving Model".format(gui.iteration))
                    pass

                dpg.add_button(
                    label="model",
                    tag="_button_save_model",
                    callback=callback_save,
                    user_data='model',
                )
                dpg.bind_item_theme("_button_save_model", theme_button)


                def callback_screenshot(sender, app_data):
                    gui.should_save_screenshot = True
                dpg.add_button(
                    label="screenshot", tag="_button_screenshot", callback=callback_screenshot
                )
                dpg.bind_item_theme("_button_screenshot", theme_button)



        # rendering options
        with dpg.collapsing_header(label="Rendering", default_open=True):
            # mode combo
            def callback_change_mode(sender, app_data):
                gui.mode = app_data
                gui.need_update = True

            dpg.add_combo(
                ("3DGS",),
                label="mode",
                default_value=gui.mode,
                callback=callback_change_mode,
            )

            # fov slider
            def callback_set_fovy(sender, app_data):
                gui.cam.fovy = np.deg2rad(app_data)
                gui.need_update = True

            dpg.add_slider_int(
                label="FoV (vertical)",
                min_value=1,
                max_value=120,
                format="%d deg",
                default_value=np.rad2deg(gui.cam.fovy),
                callback=callback_set_fovy,
            )
                

    def callback_set_mouse_loc(sender, app_data):
        if not dpg.is_item_focused("_primary_window"):
            return
        gui.mouse_loc = np.array(app_data)


    ### register camera handler

    def callback_camera_drag_rotate_or_draw_mask(sender, app_data):
        if not dpg.is_item_focused("_primary_window"):
            return

        dx = app_data[1]
        dy = app_data[2]

        gui.cam.orbit(dx, dy)
        gui.need_update = True

    def callback_camera_wheel_scale(sender, app_data):
        if not dpg.is_item_focused("_primary_window"):
            return

        delta = app_data

        gui.cam.scale(delta)
        gui.need_update = True

    def callback_camera_drag_pan(sender, app_data):
        if not dpg.is_item_focused("_primary_window"):
            return

        dx = app_data[1]
        dy = app_data[2]

        gui.cam.pan(dx, dy)
        gui.need_update = True
            
    with dpg.handler_registry():
        # for camera moving
        dpg.add_mouse_drag_handler(
            button=dpg.mvMouseButton_Left,
            callback=callback_camera_drag_rotate_or_draw_mask,
        )
        dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
        dpg.add_mouse_drag_handler(
            button=dpg.mvMouseButton_Middle, callback=callback_camera_drag_pan
        )

        dpg.add_mouse_move_handler(callback=callback_set_mouse_loc)

    dpg.create_viewport(
        title="Gaussian3D",
        width=gui.W + 600,
        height=gui.H + (45 if os.name == "nt" else 0),
        resizable=False,
    )

    ### global theme
    with dpg.theme() as theme_no_padding:
        with dpg.theme_component(dpg.mvAll):
            # set all padding to 0 to avoid scroll bar
            dpg.add_theme_style(
                dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core
            )
            dpg.add_theme_style(
                dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core
            )
            dpg.add_theme_style(
                dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core
            )

    dpg.bind_item_theme("_primary_window", theme_no_padding)

    dpg.setup_dearpygui()

    if os.path.exists("LXGWWenKai-Regular.ttf"):
        with dpg.font_registry():
            with dpg.font("LXGWWenKai-Regular.ttf", 18) as default_font:
                dpg.bind_font(default_font)

    dpg.show_viewport()

