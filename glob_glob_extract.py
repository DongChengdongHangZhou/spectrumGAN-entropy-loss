import glob
import shutil


dir_list = glob.glob('./images/*fake_B.tiff')
for i in range(7400):
    src_dir = dir_list[i]
    dst_dir = './extract/' + dir_list[i][9::]
    print(i)
    shutil.move(src_dir,dst_dir)



