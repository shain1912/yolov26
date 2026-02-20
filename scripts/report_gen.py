import pandas as pd
import os

def generate_report(defect_list, output_base="outputs/inspection_report"):
    if not defect_list:
        print("No defects found. Generating empty report.")
        df = pd.DataFrame(columns=['이미지 파일명', '하자 클래스', '위치 좌표(X, Y)', '추정 크기(px)', '심각도'])
    else:
        # 1. Convert to DataFrame
        df = pd.DataFrame(defect_list)
        
        # 2. Refine columns
        # Calculate estimate size (Area)
        df['추정 크기(px)'] = df['가로'] * df['세로']
        
        # 3. Severity Logic
        # Threshold: 10000 px^2 as an example for 'Critical'
        df['심각도'] = df['추정 크기(px)'].apply(lambda x: '높음' if x > 10000 else '보통')
        
        # Select and rename columns for report
        report_df = df[['이미지 파일명', '하자 클래스', '위치 좌표(X, Y)', '추정 크기(px)', '심각도']]
    
    # 4. Export to CSV and Excel
    csv_path = f"{output_base}.csv"
    xlsx_path = f"{output_base}.xlsx"
    
    report_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    report_df.to_excel(xlsx_path, index=False)
    
    print(f"Reports saved to {csv_path} and {xlsx_path}")
    return report_df

def generate_ai_report(report_df):
    """
    Generates a qualitative inspection report using OpenAI API based on the defect dataframe.
    """
    try:
        from openai import OpenAI
        from dotenv import load_dotenv
        
        load_dotenv()
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            try:
                import streamlit as st
                if "OPENAI_API_KEY" in st.secrets:
                    api_key = st.secrets["OPENAI_API_KEY"]
            except:
                pass
                
        client = OpenAI(api_key=api_key)
        
        # Summarize data for prompt
        total_defects = len(report_df)
        defect_counts = report_df['하자 클래스'].value_counts().to_dict()
        critical_defects = report_df[report_df['심각도'] == '높음']
        critical_count = len(critical_defects)
        
        prompt = f"""
        당신은 전문 시설물 안전 진단 전문가입니다.
        다음 드론 촬영 기반 시설물 결함 탐지 데이터를 바탕으로 상세한 안전 진단 보고서를 작성해 주세요.
        
        [탐지 요약]
        - 총 결함 수: {total_defects}개
        - 결함 유형별 분포: {defect_counts}
        - 심각한 결함(대형 균열/박락 등) 수: {critical_count}개
        
        [주요 결함값 상세 (상위 5개)]
        {report_df.head(5).to_string(index=False)}
        
        [요청 사항]
        1. 시설물의 전반적인 상태를 평가하는 **총평**을 작성하세요.
        2. 발견된 주요 결함의 위험성과 잠재적 원인을 분석하세요.
        3. 향후 필요한 유지보수 및 보강 조치를 구체적으로 제안하세요.
        4. 보고서 톤은 전문적이고 객관적으로 유지해 주세요.
        """
        
        response = client.chat.completions.create(
            model="gpt-5-nano",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for structural safety inspection."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"AI 보고서 생성 실패: {str(e)}"

if __name__ == "__main__":
    # Test data
    mock_data = [
        {"이미지 파일명": "test1.jpg", "하자 클래스": "crack", "위치 좌표(X, Y)": (100, 200), "가로": 50, "세로": 300},
        {"이미지 파일명": "test1.jpg", "하자 클래스": "spalling", "위치 좌표(X, Y)": (400, 500), "가로": 200, "세로": 200},
    ]
    generate_report(mock_data)
