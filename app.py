import streamlit as st
import cv2
import pandas as pd
import os
from PIL import Image
from scripts.inference import run_inference
from scripts.report_gen import generate_report, generate_ai_report

st.set_page_config(layout="wide", page_title="Drone Defect Detection PoC")

st.title("🚁 드론 영상 기반 시설물 하자 자동 탐지 AI 솔루션")
st.markdown("---")

# Sidebar for Model Selection/Status
st.sidebar.header("시스템 설정")
model_path = st.sidebar.text_input("모델 경로", "models/best.pt")
status_placeholder = st.sidebar.empty()

# Image Upload
uploaded_file = st.file_uploader("드론 촬영 이미지를 업로드하세요", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Reset AI report state if image changes
    if 'last_uploaded_file' not in st.session_state or st.session_state['last_uploaded_file'] != uploaded_file.name:
        st.session_state['last_uploaded_file'] = uploaded_file.name
        if 'ai_report' in st.session_state:
            del st.session_state['ai_report']

    # 1. Save uploaded file temporarily
    temp_path = os.path.join("data/images/val", uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # 2. Run Inference
    with st.spinner("AI가 하자를 분석 중입니다..."):
        try:
            defect_data = run_inference(temp_path, model_path)
            report_df = generate_report(defect_data)
        except Exception as e:
            st.error(f"분석 중 오류 발생: {e}")
            defect_data = None

    # 3. Layout: Image | Data Table
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("탐지 결과 이미지")
        if os.path.exists("outputs/result_img.jpg"):
            res_img = Image.open("outputs/result_img.jpg")
            st.image(res_img, use_container_width=True)
        else:
            st.warning("결과 이미지를 생성할 수 없습니다.")

    with col2:
        st.subheader("결함 분석 데이터")
        if defect_data:
            st.dataframe(report_df, use_container_width=True)
            
            # 4. Download Button
            csv = report_df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="리포트 CSV 다운로드",
                data=csv,
                file_name='inspection_report.csv',
                mime='text/csv',
            )
            
            # 5. OpenAI Report Generation
            st.markdown("---")
            st.subheader("🤖 AI 안전 진단 보고서")
            if st.button("AI 보고서 생성하기"):
                with st.spinner("AI가 보고서를 작성 중입니다..."):
                    ai_report = generate_ai_report(report_df)
                    st.session_state['ai_report'] = ai_report
            
            if 'ai_report' in st.session_state:
                st.markdown(st.session_state['ai_report'])
                st.download_button(
                    label="AI 보고서 다운로드 (Markdown)",
                    data=st.session_state['ai_report'],
                    file_name='ai_inspection_report.md',
                    mime='text/markdown'
                )
                    
        else:
            st.write("탐지된 하자가 없습니다.")

else:
    st.info("이미지를 업로드하면 AI가 균열, 박락, 철근노출 여부를 판단합니다.")
    # Show example layout
    st.image("https://via.placeholder.com/1200x500.png?text=Drone+PoC+Dashboard+Preview", use_container_width=True)
