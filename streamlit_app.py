import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import datetime
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from face_alignment import FaceAlignment, LandmarksType
import numpy as np

# 거리 계산 함수
def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# 얼굴 분석 함수
def analyze_face(landmarks):
    result = {}

    symmetry_indices = [
        (17, 26), (18, 25), (19, 24), (20, 23), (21, 22),
        (36, 45), (37, 44), (38, 43), (39, 42), (40, 47), (41, 46),
        (31, 35), (32, 34), (48, 54), (49, 53), (50, 52),
        (3, 13), (4, 12)
    ]
    diffs = [euclidean_distance(landmarks[i], landmarks[j]) for i, j in symmetry_indices]
    result["symmetry_score"] = np.mean(diffs)

    result["left_eye_width"] = euclidean_distance(landmarks[36], landmarks[39])
    result["left_eye_height"] = euclidean_distance(landmarks[37], landmarks[41])
    result["right_eye_width"] = euclidean_distance(landmarks[42], landmarks[45])
    result["right_eye_height"] = euclidean_distance(landmarks[43], landmarks[47])

    eye_dx = landmarks[45][0] - landmarks[36][0]
    eye_dy = landmarks[45][1] - landmarks[36][1]
    result["roll_diff"] = np.degrees(np.arctan2(eye_dy, eye_dx))

    result["eye_inner_dist"] = euclidean_distance(landmarks[39], landmarks[42])
    result["philtrum_length"] = euclidean_distance(landmarks[33], landmarks[51])
    result["nose_length"] = euclidean_distance(landmarks[27], landmarks[33])
    result["nose_width"] = euclidean_distance(landmarks[31], landmarks[35])
    forehead_point = landmarks[27] + (landmarks[27] - landmarks[8]) * 0.25
    brow_center = (landmarks[21] + landmarks[22]) / 2
    result["forehead_height"] = euclidean_distance(forehead_point, brow_center)
    result["chin_lip_length"] = euclidean_distance(landmarks[57], landmarks[8])
    result["face_length"] = euclidean_distance(forehead_point, landmarks[8])
    result["face_width"] = euclidean_distance(landmarks[1], landmarks[15])

    return result

# 얼굴 분석 결과를 이미지에 시각화하는 함수
def draw_landmark_overlay(image, landmarks, results):
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)

    def to_point(p):
        return tuple(map(int, p))

    # 숫자 라벨용 폰트 (기본)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except:
        font = ImageFont.load_default()

    # 각 측정마다 다른 선 색상 지정
    LINE_COLORS = [
        "red", "blue", "green", "orange", "purple", "cyan", "magenta",
        "gold", "deepskyblue", "salmon", "lime", "violet", "darkorange",
        "dodgerblue", "mediumseagreen", "tomato", "plum"
    ]
    TEXT_COLOR = "white"

    # 6. 왼쪽 눈 너비
    draw.line([to_point(landmarks[36]), to_point(landmarks[39])], fill=LINE_COLORS[0], width=2)
    draw.text(to_point((np.array(landmarks[36]) + np.array(landmarks[39])) / 2 + np.array([0, -10])), "6", fill=TEXT_COLOR, font=font)

    # 7. 왼쪽 눈 높이
    draw.line([to_point(landmarks[37]), to_point(landmarks[41])], fill=LINE_COLORS[1], width=2)
    draw.text(to_point((np.array(landmarks[37]) + np.array(landmarks[41])) / 2 + np.array([5, 0])), "7", fill=TEXT_COLOR, font=font)

    # 8. 오른쪽 눈 너비
    draw.line([to_point(landmarks[42]), to_point(landmarks[45])], fill=LINE_COLORS[2], width=2)
    draw.text(to_point((np.array(landmarks[42]) + np.array(landmarks[45])) / 2 + np.array([0, -10])), "8", fill=TEXT_COLOR, font=font)

    # 9. 오른쪽 눈 높이
    draw.line([to_point(landmarks[43]), to_point(landmarks[47])], fill=LINE_COLORS[3], width=2)
    draw.text(to_point((np.array(landmarks[43]) + np.array(landmarks[47])) / 2 + np.array([5, 0])), "9", fill=TEXT_COLOR, font=font)

    # 10. 미간 거리
    draw.line([to_point(landmarks[39]), to_point(landmarks[42])], fill=LINE_COLORS[4], width=2)
    draw.text(to_point((np.array(landmarks[39]) + np.array(landmarks[42])) / 2 + np.array([0, -10])), "10", fill=TEXT_COLOR, font=font)

    # 11. 인중
    draw.line([to_point(landmarks[33]), to_point(landmarks[51])], fill=LINE_COLORS[5], width=2)
    draw.text(to_point((np.array(landmarks[33]) + np.array(landmarks[51])) / 2 + np.array([5, 0])), "11", fill=TEXT_COLOR, font=font)

    # 12. 코 길이
    draw.line([to_point(landmarks[27]), to_point(landmarks[33])], fill=LINE_COLORS[6], width=2)
    draw.text(to_point((np.array(landmarks[27]) + np.array(landmarks[33])) / 2 + np.array([5, 0])), "12", fill=TEXT_COLOR, font=font)

    # 13. 코 너비
    draw.line([to_point(landmarks[31]), to_point(landmarks[35])], fill=LINE_COLORS[7], width=2)
    draw.text(to_point((np.array(landmarks[31]) + np.array(landmarks[35])) / 2 + np.array([0, -10])), "13", fill=TEXT_COLOR, font=font)

    # 14. 이마 높이
    forehead_point = landmarks[27] + (landmarks[27] - landmarks[8]) * 0.25
    brow_center = (landmarks[21] + landmarks[22]) / 2
    draw.line([to_point(forehead_point), to_point(brow_center)], fill=LINE_COLORS[8], width=2)
    draw.text(to_point((forehead_point + brow_center) / 2 + np.array([5, 0])), "14", fill=TEXT_COLOR, font=font)

    # 15. 입~턱 길이
    draw.line([to_point(landmarks[57]), to_point(landmarks[8])], fill=LINE_COLORS[9], width=2)
    draw.text(to_point((np.array(landmarks[57]) + np.array(landmarks[8])) / 2 + np.array([5, 0])), "15", fill=TEXT_COLOR, font=font)

    # 16. 얼굴 세로 길이
    forehead_top = landmarks[27] + (landmarks[27] - landmarks[8]) * 0.6
    draw.line([to_point(forehead_top), to_point(landmarks[8])], fill=LINE_COLORS[10], width=2)
    draw.text(to_point((forehead_top + landmarks[8]) / 2 + np.array([-10, 0])), "16", fill=TEXT_COLOR, font=font)

    # 17. 얼굴 가로 길이
    draw.line([to_point(landmarks[1]), to_point(landmarks[15])], fill=LINE_COLORS[11], width=2)
    draw.text(to_point((np.array(landmarks[1]) + np.array(landmarks[15])) / 2 + np.array([0, -10])), "17", fill=TEXT_COLOR, font=font)

    # 만족도 표시
    if "satisfaction" in results:
        draw.text((10, 10), f"Satisfaction: {results['satisfaction']}", fill=TEXT_COLOR, font=font)

    return image



# ✅ 구글 시트 인증
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("selfie-analyze-770dd0771bc0.json", scope)
client = gspread.authorize(creds)

# ✅ 오늘 날짜 기준으로 시트 열기
today = datetime.date.today().strftime("%Y-%m-%d")
sheet_name = f"셀카분석_결과_250421"
worksheet = client.open(sheet_name).sheet1

st.title("📷 셀카 AI 분석기")

# ✅기본값 설정
default_values = {
    "id_num": "",
    "angle": 0.0,
    "satisfaction": 5,
    "reason": ""
}

# ✅ 초기화 버튼
if st.button("입력 초기화", key="reset_button"):
    for key, val in default_values.items():  # uploaded_file은 여기 없음!
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

# ✅ 입력창: 기본값은 session_state에서 가져오기
st.subheader("1. 기본 정보 입력")
id_num = st.text_input("학번(5자리숫자)", value=st.session_state.get("id_num", ""), key="id_num")
angle = st.number_input(
    "촬영 각도 (위 = +, 아래 = - )",
    min_value=-90.0,
    max_value=90.0,
    step=0.1,
    format="%.1f",
    value=st.session_state.get("angle", 0.0),  # ✅ 명시적 기본값 지정
    key="angle",
    help="예: 아래에서 찍었으면 -30.0, 위에서 찍었으면 +20.0"
)

satisfaction = st.number_input(
    "셀카 만족도 점수 (1.0 ~ 10.0)",
    min_value=1.0,
    max_value=10.0,
    step=0.1,
    format="%.1f",
    value=st.session_state.get("satisfaction", 5.0),  # ✅ 명시적 기본값 지정
    key="satisfaction"
)

reason = st.text_area(
    "왜 이 셀카가 마음에 들었나요?",
    value=st.session_state.get("reason", ""),  # ✅ 명시적 기본값 지정
    key="reason"
)
st.markdown("💡 사진 업로드 후 **잠시 기다리면** 분석 결과가 아래에 나옵니다!")

st.subheader("2. 셀카 사진 업로드")
uploaded_file = st.file_uploader("이미지 파일을 업로드하세요", type=["jpg", "jpeg", "png"])

# 🔹 사진 업로드
if uploaded_file:
    # 얼굴 분석기 초기화는 여기서 실행!
    try:
        fa = FaceAlignment(LandmarksType.TWO_D, device='cpu')
    except Exception as e:
        st.error(f"얼굴 분석 모델을 불러오는 데 실패했습니다: {e}")
        st.stop()

    # 이미지 분석 시작작
    image = Image.open(uploaded_file)
    st.image(image, caption="업로드된 셀카", use_column_width=True)

    # 얼굴 특징점 추출
    np_image = np.array(image.convert("RGB"))
    preds = fa.get_landmarks(np_image)

    if preds is None:
        st.warning("얼굴을 감지할 수 없습니다. 다시 촬영해 주세요.")
    else:
        landmarks = preds[0]
        results = analyze_face(landmarks)  # 👈 분석 실행
        results["satisfaction"] = satisfaction  # 만족도도 결과에 포함시킴

        # 분석 시각화 이미지 만들기
        annotated_image = draw_landmark_overlay(image, landmarks, results)

        # Streamlit에 표시
        st.subheader("3. 분석 결과 시각화")
        st.image(annotated_image, caption="📊 얼굴 분석 위치 보기", use_column_width=True)

        # 분석 결과 표시
        st.subheader("3. 분석 결과")
        st.write(results)

        if st.button("제출하기"):
            # 학번이 비었거나, 5자리 숫자가 아니면
            if not id_num or not id_num.isdigit() or len(id_num) != 5:
                st.warning("⚠️ 학번은 반드시 5자리 숫자여야 합니다.")
            elif angle is None or satisfaction is None or reason.strip() == "":
                st.warning("⚠️ 촬영각도, 만족도, 만족도 이유를 모두 입력해주세요.")
            elif uploaded_file is None:
                st.warning("⚠️ 셀카 사진도 업로드해주세요.")
            else:
                def to_float(x):
                    return float(x) if isinstance(x, (np.float32, np.float64, np.ndarray)) else x

                worksheet.append_row([
                    id_num, angle, satisfaction,
                    to_float(results["symmetry_score"]), to_float(results["roll_diff"]),
                    to_float(results["left_eye_width"]), to_float(results["left_eye_height"]),
                    to_float(results["right_eye_width"]), to_float(results["right_eye_height"]),
                    to_float(results["eye_inner_dist"]), to_float(results["philtrum_length"]),
                    to_float(results["nose_length"]), to_float(results["nose_width"]),
                    to_float(results["forehead_height"]), to_float(results["chin_lip_length"]),
                    to_float(results["face_length"]), to_float(results["face_width"]),
                    reason
                ])

                st.success("제출 완료! 감사합니다 😊 맨 위  입력 초기화 버튼을 클릭하여 다른 사진도 분석해서 제출해보세요!")
                st.info("💡 입력 초기화 시 사진은 유지됩니다. 다른 사진을 분석하려면 다시 업로드해주세요.")

   
