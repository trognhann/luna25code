import numpy as np
import SimpleITK as sitk


def extract_nodule_with_metadata(mha_path, center_world_coord, target_shape=(64, 128, 128)):
    """
    Trích xuất nodule từ file mha dựa trên tọa độ thực và shape mong muốn.

    Args:
        mha_path (str): Đường dẫn đến file .mha
        center_world_coord (tuple): Tọa độ (x, y, z) thực tế của tâm nodule (mm).
        target_shape (tuple): Kích thước đầu ra mong muốn (Depth, Height, Width).
                              Ví dụ: (64, 128, 128) tương ứng (Z, Y, X).

    Returns:
        numpy_array: Nodule đã cắt với shape target_shape.
    """
    # 1. Đọc file MHA
    itk_image = sitk.ReadImage(mha_path)

    # --- KIỂM TRA METADATA (So khớp với dữ liệu bạn cung cấp) ---
    # SimpleITK trả về Spacing theo thứ tự (x, y, z)
    # Numpy array có shape (z, y, x)
    spacing = itk_image.GetSpacing()
    origin = itk_image.GetOrigin()

    print(f"--- Metadata từ file {mha_path} ---")
    print(f"Origin: {origin}")   # Nên khớp với [-238.07, ...]
    print(f"Spacing: {spacing}") # Nên khớp với [0.705, 0.705, 2.0] (ITK đảo thứ tự so với numpy ZYX)

    # 2. Chuyển đổi Tọa độ World (mm) -> Voxel Index
    # SimpleITK tự động dùng origin và spacing trong file để tính
    center_idx = itk_image.TransformPhysicalPointToIndex(center_world_coord)
    print(f"Tâm khối u (World): {center_world_coord}")
    print(f"Tâm khối u (Index - x,y,z): {center_idx}")

    # 3. Lấy toàn bộ mảng ảnh (Thứ tự shape lúc này là Z, Y, X)
    full_image_arr = sitk.GetArrayFromImage(itk_image)

    # 4. Tính toán vùng cắt (Bounding Box)
    # target_shape = (D, H, W) tương ứng (Z, Y, X)
    d_z, d_y, d_x = target_shape

    # SimpleITK center_idx là (x, y, z) -> Numpy cần (z, y, x)
    c_x, c_y, c_z = center_idx

    # Tính điểm bắt đầu (Start) và kết thúc (End)
    z_start = c_z - (d_z // 2)
    y_start = c_y - (d_y // 2)
    x_start = c_x - (d_x // 2)

    z_end = z_start + d_z
    y_end = y_start + d_y
    x_end = x_start + d_x

    # 5. Xử lý Padding (Nếu vùng cắt bị ra ngoài biên ảnh)
    # Tạo array chứa giá trị nền (Air = -1000 HU)
    extracted_nodule = np.full(target_shape, -1000.0, dtype=np.float32)

    img_z, img_y, img_x = full_image_arr.shape

    # Tìm vùng giao nhau (Intersection) giữa vùng muốn cắt và ảnh gốc
    # Giới hạn tọa độ trong ảnh gốc
    z_s_lim = max(0, z_start); z_e_lim = min(img_z, z_end)
    y_s_lim = max(0, y_start); y_e_lim = min(img_y, y_end)
    x_s_lim = max(0, x_start); x_e_lim = min(img_x, x_end)

    # Nếu vùng giao hợp lệ
    if (z_e_lim > z_s_lim) and (y_e_lim > y_s_lim) and (x_e_lim > x_s_lim):
        # Tính toán vị trí tương ứng trong khối đích (extracted_nodule)
        out_z_s = z_s_lim - z_start; out_z_e = out_z_s + (z_e_lim - z_s_lim)
        out_y_s = y_s_lim - y_start; out_y_e = out_y_s + (y_e_lim - y_s_lim)
        out_x_s = x_s_lim - x_start; out_x_e = out_x_s + (x_e_lim - x_s_lim)

        # Copy dữ liệu
        extracted_nodule[out_z_s:out_z_e, out_y_s:out_y_e, out_x_s:out_x_e] = \
            full_image_arr[z_s_lim:z_e_lim, y_s_lim:y_e_lim, x_s_lim:x_e_lim]

    print(f"Shape đã extract: {extracted_nodule.shape}")
    return extracted_nodule