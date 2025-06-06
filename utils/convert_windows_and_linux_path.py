import os
# 示例映射
base_mapping_linux_to_windows = {
    '/media/weifeng/移动硬盘':  'G:',
    '/media/weifeng/1号硬盘-weifeng':'F:'
}

base_mapping_windows_to_linux = {
    'G:':'/media/weifeng/移动硬盘',
    'F:':'/media/weifeng/1号硬盘-weifeng'

}



def linux_to_windows_path(linux_path, base_mapping):

    # 将Linux路径转换为Windows路径。

    # :param linux_path: Linux路径（如 /home/user/documents）
    # :param base_mapping: 映射字典，键为Linux根路径，值为Windows对应路径（如 {'/home': 'C:\\Users\\User'}）
    # :return: 转换后的Windows路径

    for linux_base, windows_base in base_mapping.items():
        if linux_path.startswith(linux_base):
            # 替换路径
            windows_path = linux_path.replace(linux_base, windows_base, 1)
            # return windows_path.replace('/', '\\')  # 替换正斜杠为反斜杠
            return windows_path.replace('\\', '/')  # 替换反斜杠为正斜杠

    # 如果没有找到匹配的映射，返回原始路径
    return linux_path.replace('\\', '/')


def windows_to_linux_path(windows_path, base_mapping):

    # 将Windows路径转换为Linux路径。

    # :param windows_path: Windows路径（如 C:\Users\User\Documents）
    # :param base_mapping: 映射字典，键为Windows根路径，值为Linux对应路径（如 {'C:\\Users\\User': '/home'}）
    # :return: 转换后的Linux路径

    for windows_base, linux_base in base_mapping.items():
        if windows_path.startswith(windows_base):
            # 替换路径
            linux_path = windows_path.replace(windows_base, linux_base, 1)
            return linux_path.replace('\\', '/')  # 替换反斜杠为正斜杠

    # 如果没有找到匹配的映射，返回原始路径
    return windows_path.replace('\\', '/')


def convert_paths(old_path):
    if os.name == 'nt':
        # print("当前运行在Windows平台")
        linux_path=old_path
        converted_windows_path = linux_to_windows_path(linux_path, base_mapping_linux_to_windows)
        # print("对应的Windows路径:", converted_windows_path)
        return converted_windows_path
    elif os.name == 'posix':
        # print("当前运行在Linux平台或其他类Unix平台")
        windows_path= old_path
        converted_linux_path = windows_to_linux_path(windows_path, base_mapping_windows_to_linux)
        # print("对应的Linux路径:", converted_linux_path)
        return converted_linux_path
    else:
        print("未知平台")
        return old_path
    pass

# # 示例
# linux_path = '/home/user/documents'
# windows_path = 'G:\\Users\\User\\Documents'
#
# convert_paths(old_path=linux_path)
# convert_paths(old_path=windows_path)



