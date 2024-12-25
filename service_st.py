import streamlit as st
import torch
from pit_extractor import PiTExtractor
from PIL import Image
import tempfile
from detector import Detector

@st.cache_resource()
def create_model(device):
    extractor = PiTExtractor(device=device)
    detector = Detector(device=device)
    return extractor, detector


@torch.inference_mode()
def main(
    device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    torch_device = torch.device(device)

    look_num = 5

    extractor, detector = create_model(device=torch_device)

    args_contrainer = st.container(border=True)
    with args_contrainer:
        look_num = args_contrainer.number_input("visualize number", min_value=1, max_value=20, value=5)
        use_detector = args_contrainer.checkbox("Use detector", value=False, )

    image_or_video = st.file_uploader("Input", type=["jpg", "JPEG", "png", "webp", "mp4"])
    # save temp file in the server
    if image_or_video is not None:
        if image_or_video.type.split("/")[0] == "image":
            image = image_or_video
            with tempfile.NamedTemporaryFile(delete=False) as fp:
                fp.write(image.getvalue())
                image = Image.open(fp.name).convert("RGB")

                # select the look num

                if st.button("Query"):
                    if use_detector:
                        boxes = detector.detect(fp.name)
                        if len(boxes) == 0:
                            st.write("No object detected")
                            return
                        cropped_images = detector.crop_image(image, boxes)
                        confidences = boxes[0]["confidences"]
                        
                        for i, cropped_image in enumerate(cropped_images):
                            rank, rank_dist, rank_file, rank_feat = extractor.query(cropped_image, look_num=look_num)
                            contrainer = st.container(border=True)
                            contrainer.write("Query image")
                            contrainer.image(cropped_image, caption=f"detector confidence: {confidences[i]:.2f}")

                            contrainer.write("neighbor images in the database")

                            for i in range(0, len(rank), 5):
                                row = contrainer.columns(5)
                                for j, col in enumerate(row):
                                    col.image(rank_file[i + j], caption=f"Distance: {rank_dist[i + j]:.2f}")
                        
                        

                    else:
                        rank, rank_dist, rank_file, rank_feat = extractor.query(fp.name, look_num=look_num)
                        
                        # Display the results including image and distance and its rank in the database
                        # 排列顺序是从小到大，所以越小越相似， 按照每行5个图片的方式展示


                        contrainer = st.container(border=True)
                        contrainer.write("Query image")
                        contrainer.image(fp.name)

                        contrainer.write("neighbor images in the database")

                        for i in range(0, len(rank), 5):
                            row = contrainer.columns(5)
                            for j, col in enumerate(row):
                                col.image(rank_file[i + j], caption=f"Distance: {rank_dist[i + j]:.2f}")

                            
        elif image_or_video.type.split("/")[0] == "video":
            video = image_or_video
            if use_detector:
                st.error("Detector is not supported for video")
            with tempfile.NamedTemporaryFile(delete=False) as fp:
                fp.write(video.getvalue())
                video = fp.name
                if st.button("Query"):
                    rank, rank_dist, rank_file, rank_feat = extractor.query_video(video, look_num=look_num)
                    
                    # Display the results including image and distance and its rank in the database
                    # 排列顺序是从小到大，所以越小越相似， 按照每行5个图片的方式展示

                    contrainer = st.container(border=True)
                    contrainer.write("Query video")
                    contrainer.video(video)

                    contrainer.write("neighbor images in the database")

                    for i in range(0, len(rank), 5):
                        row = contrainer.columns(5)
                        for j, col in enumerate(row):
                            col.image(rank_file[i + j], caption=f"Distance: {rank_dist[i + j]:.2f}")
                
    else:
        st.write("Please upload an image or video")


if __name__ == "__main__":
    main()
    