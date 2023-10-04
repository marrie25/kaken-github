#2023.08.28 test2
import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import Voronoi, voronoi_plot_2d
import os
import csv

#program->
def scale(eigen_img, s):
        scale_percent = s
        width = int(eigen_img.shape[1] * scale_percent / 100)
        height = int(eigen_img.shape[0] * scale_percent / 100)
        dim = (width, height)
        eigen_img = cv2.resize(eigen_img, dim)

        return eigen_img

def ready(uploaded_file, image_size, PATH): #読み込み+画像の縮尺
    if uploaded_file is None:
        st.write("Set the file name...(ファイル名をアップロードしてください。)")
        return

    imagename=uploaded_file.name
    #PATH = os.getcwd()
    #st.write(PATH + "/" + imagename)

    #st.write(PATH)
    #st.write(os.listdir(PATH + "/" +"download" ))

    if os.path.isfile(PATH+"/"+ imagename):
        st.write("file OK")

    #eigen_img = cv2.imread(PATH+"/"+ imagename)
    #eigen_img = cv2.imread(imagename)

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()),dtype=np.uint8)
        eigen_img = cv2.imdecode(file_bytes, 1)


    if eigen_img is None:
        exit(-1)
    harris_img = eigen_img.copy()
    fast_img = eigen_img.copy()

    b = image_size
    s = int(b)

    eigen_img=scale(eigen_img, s)

    #img_flip_ud_lr=cv2.flip(eigen_img, -1)
    img_flip_ud_lr = eigen_img
    # cv2.imwrite(img_flip_ud_lr)

    return img_flip_ud_lr


def generate_binary(img_flip_ud_lr): #二値化
    if img_flip_ud_lr is None:
        print("Error:img_flip_ud_lr is None")
        return

    gray_img = cv2.cvtColor(img_flip_ud_lr, cv2.COLOR_BGR2GRAY)
    ret, img_thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #cv2.imwrite(binary_output,img_thresh)

    st.image(img_thresh)
    return gray_img, img_thresh


def generate_point(img_flip_ud_lr): #特徴点抽出
    if img_flip_ud_lr is not None:
        gray_img=cv2.cvtColor(img_flip_ud_lr, cv2.COLOR_BGR2GRAY)
        img_thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        num_points=feature
        #img_thresh=np.array(img_thresh)

        num_feature_points = cv2.goodFeaturesToTrack(gray_img, int(num_points), 0.01, 5)

        st.write("Number of Feature Points:", len(num_feature_points))
        #st.write("Feature Points:", num_feature_points)

        if num_feature_points is not None:
            num_feature_points = np.intp(num_feature_points)

            for num_feature_point in num_feature_points:
                x, y = num_feature_point.ravel()
                cv2.circle(gray_img, (x, y), 1, (0, 200, 255), -1)
                cv2.circle(gray_img, (x, y), 1, (0, 200, 255), 1)


            st.image(gray_img)

        d=st.text_input("Set the file name...(ファイル名を決めてください)") #画像出力パス
        image_output=d
        st.caption("file name.jpg or png or jpeg(ファイル名.jpeg )")

        if st.button("Save Image"):
            if not image_output.strip():
                st.write("Please enter a valid image file name.")
            else:
                # Convert the image to bytes
                image_bytes = cv2.imencode('.jpg', img_flip_ud_lr)[1].tobytes()

                # Provide the image data for download
                st.download_button(
                    label="Click to download",
                    data=image_bytes,
                    file_name=image_output,
                    key="download_image_button"
                )
       
        output_path=d        
        
        #cv2.imwrite(output_path, img_flip_ud_lr)

        csv_out =st.text_input("Set the csv file name...(csvファイル名を決めてください)") #csvパス
        st.caption("file name.csv(ファイル名.csv)")

        output_path=csv_out        
        #output_path=csv_output

        

        if csv_out.strip()=="":
            st.write("Please enter a valid CSV file path.")


        else:
            try:
                with open(output_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['x座標', 'y座標'])
                    for point in num_feature_points:
                        x, y = point.ravel()
                        writer.writerow([x, y])
                 
                    #df1=pd.read_csv(output_path, encoding='utf-8')
                    #st.table(df1)

                    return output_path

            except Exception as e:
                    st.write(f"Error: {str(e)}")


def generate_voronoi_diagram(output_path): #ボロノイ図作成 -with Voronoi module-
    #st.write("Path is ",  output_path)
    if  output_path is None:
        st.write("Output path is not provided. Please set a valid CSV file path.")
        return

    try:
        df = pd.read_csv(output_path)
        plt.scatter(df['x座標'], df['y座標'], s=0.3)
        df['y座標'] = df['y座標'].max() - df['y座標']
        points = np.array(list(zip(df['x座標'], df['y座標'])))[::-1]

        vor = Voronoi(points)
        
        plt.figure()
        fin_voro = voronoi_plot_2d(vor)

        st.pyplot(fin_voro)

    except Exception as e:
        st.write(f"Error: {str(e)}")
        st.write("Please make sure the provided CSV file path is correct.")







# setup working directory
PATH = os.getcwd()
output_path=None

#サイト
st.title("Vorography") #title

st.markdown(""" <style> .font {
    font-0size:35px ; font-family: 'Cooper Black'; color: #ffffff;}
    </style> """, unsafe_allow_html=True)
st.markdown('<p class="font">Upload your photo here...</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("file Upload", type=(['jpeg']))  #画像のアップロード
image_size = st.text_input("Set the size of the image...(画像の大きさを入力してください)") #画像のサイズ
st.caption("Set the scale as a percentage  ex:80%->80(縮尺をパーセンテージで入れてください 例:80%->80)")#入力例を入れる
#image_output =st.text_input("Set the file name...(ファイル名を決めてください)") #画像出力パス
#st.caption("file name.jpg or png or jpeg(ファイル名.jpg or png or jpeg )")
#csv_output =st.text_input("Set the csv file name...(csvファイル名を決めてください)") #csvパス
#st.caption("file name.csv(ファイル名.csv)")

switch1=st.checkbox("Next")

if switch1:
    img_flip_ud_lr=ready(uploaded_file, image_size, PATH)
    if img_flip_ud_lr is not None:
        st.image(img_flip_ud_lr)


    #tab1, tab2= st.tabs(["Point(特徴点)", "Binarization(二値化)"])

    #with tab2:
        st.header("Binarization(二値化)")
        generate_binary(img_flip_ud_lr)


    #with tab1:
        st.header("Features point extraction(特徴点抽出)")
        feature=st.slider("Determine the number of points...(点の数を決めてください)", 0 ,1000 ,50) # スライダー
        num_feature_points=feature
        st.caption("Once you have determined the number of dots, be sure to check the box(点の数を決めたら、必ずチェックを入れて下さい)")
        switch2=st.checkbox("Done")
        if switch2:
            output_path=generate_point(img_flip_ud_lr)

            switch3=st.button("Next")
            if output_path is not None:
                st.caption("(ボタンを押したら、ボロノイ図が生成されます)")
            
                if switch3:
                    generate_voronoi_diagram(output_path)
                    

                    #switch4=st.button("Coloring")
                    #if switch4:
                    

                    #    tab1 = st.tabs(["Coloring(着色)"])
                    #    tab2 = st.tabs(["(重ね合わせ)"])
                    
                        #with tab2:
                        #    st.header("(重ね合わせ)")
                        #    #st.image()

                    #    with tab1:
                    #        st.header("Coloring(着色)")

                    #        st.subheader("Color sample(色見本)")
                        #色見本の表示
                    #        data={'color':['Red(赤)', 'Reddish-purple(赤紫)', 'Pink(ピンク)', 'Purple(紫)', 'Blue(青)', 'Bluish-purple(青紫)', 'Aqua(水色)', 'Bluish-green(青緑)', 'Green(緑)', 'Yellow-green(黄緑)', 'Yellow(黄色)', 'Orange(オレンジ)', 'Brown(茶色)', 'Black(黒色)', 'Grey(灰色)'],
                    #        'Red':['230', '235', '245', '136', '0', '103', '188', '0', '62', '184', '255', '238', '150', '43', '125'],
                    #        'Green' :['0', '110', '178', '72', '149', '69', '226', '164', '179', '210', '217', '120', '80', '43', '125'],
                    #        'Blue' :['51', '165', '178', '152', '217', '60','232', '151', '112', '0', '0', '0', '66', '43', '125']
                    #        }


                    #        df = pd.DataFrame(data)
                    #        st.table(df) # 静的な表

                    #        st.caption("Colors other than those in the color samples can be used.(色見本以外の色も使用可能です)")

                    #        r=st.slider("choose red",0,256,128,1)
                    #        g=st.slider("choose green",0,256,128,1)
                    #        b=st.slider("choose blue",0,256,128,1)

                    #        st.caption("Once you decide on a color, be sure to check the box!(色を決めたら、チェックして下さい)")
                    #        check3=st.checkbox("Done")
                    #    if check3:
                    #        generate_color(points, xlim, ylim, region_colors)
                    #        st.pyplot()

                        #tab3 = st.tabs("Area(面積)")
                        #with tab3:
                        #    st.header("Area")