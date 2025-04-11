import streamlit as st
st.set_page_config(page_title="부산 기업 RAG", layout="wide")

import os
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document, ChatResult
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema.messages import BaseMessage, HumanMessage, AIMessage
from langchain.chat_models.base import BaseChatModel
from groq import Groq

# ✅ 커스텀 ChatModel 클래스
class GroqLlamaChat(BaseChatModel):
    groq_api_key: str
    model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    _client: Groq = None

    def __init__(self, **data):
        super().__init__(**data)
        self._client = Groq(api_key=self.groq_api_key)

    def _call(self, messages, **kwargs):
        formatted = []
        for m in messages:
            if isinstance(m, HumanMessage):
                formatted.append({"role": "user", "content": m.content})
            elif isinstance(m, AIMessage):
                formatted.append({"role": "assistant", "content": m.content})
        response = self._client.chat.completions.create(
            model=self.model,
            messages=formatted,
        )
        return response.choices[0].message.content

    def _generate(self, messages: list[BaseMessage], stop=None, **kwargs) -> ChatResult:
        content = self._call(messages, **kwargs)
        return ChatResult(
            generations=[{"text": content, "message": AIMessage(content=content)}]
        )

    @property
    def _llm_type(self):
        return "groq-llama-4"

    @property
    def _identifying_params(self):
        return {"model": self.model}

# ✅ 텍스트 파일 로딩 함수
def load_api_key():
    api_key = st.secrets["general"]["API_KEY"]
    return api_key

def load_template():
    with open("template.txt", "r", encoding="utf-8") as file:
        return file.read()

# ✅ 초기 컴포넌트 캐싱 (1회만 실행)
@st.cache_resource
def init_qa_chain():
    api_key = load_api_key()
    template = load_template()

    # 임베딩 모델
    embedding_model = HuggingFaceEmbeddings(model_name="jhgan/ko-sbert-nli")

    # 벡터 스토어
    vectorstore = FAISS.load_local("busan_db", embedding_model, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # LLM & 프롬프트
    llm = GroqLlamaChat(groq_api_key=api_key)
    prompt = PromptTemplate.from_template(template)

    # QA 체인 생성
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )
    return qa_chain

# ✅ QA 체인 세션 상태에 저장
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = init_qa_chain()

# ✅ UI 구성
st.title("🚢 부산 취업 상담 챗봇(JOB MAN)")

query = st.text_input("🎯 질문을 입력하세요:", placeholder="예) 신입 사원이 처음 받는 연봉 3000만원 이상 되는 선박 제조업 회사를 추천해줘")

# ✅ 버튼 클릭 시, 체인 실행만!
if st.button("💬 질문 실행") and query:
    with st.spinner("🤖 JOB MAN이 부산 기업 정보를 검색 중입니다..."):
        result = st.session_state.qa_chain.invoke(query)

        st.subheader("✅ JOB MAN의 답변")
        st.write(result["result"])

        st.subheader("📚 참고 문서")
        for i, doc in enumerate(result["source_documents"]):
            with st.expander(f"문서 {i+1}"):
                st.write(doc.page_content)


import pandas as pd
import streamlit as st
from geopy.geocoders import GoogleV3
import folium

# Google API 키 입력 (발급받은 API 키로 교체)
google_api_key = "YOUR_GOOGLE_API_KEY"  # Google API 키 입력

# GoogleV3 Geocoder 설정
geolocator = GoogleV3(api_key=google_api_key)

# CSV 파일 경로에 맞게 수정
df = pd.read_csv("부산광역시_제조업 공장등록 현황_241231 (1).csv", encoding='cp949')  # 또는 'euc-kr'

# 스트림릿 애플리케이션 제목
st.title("회사 위치 지도")

# 지도 생성: 부산의 기본 좌표로 설정
map = folium.Map(location=[35.1796, 129.0756], zoom_start=12)

# 각 회사에 대해 주소를 위도와 경도로 변환하고 마커 추가
for index, row in df.iterrows():
    address = row['공장대표주소(지번)']
    company_name = row['회사명']
    
    try:
        # 주소를 위도, 경도로 변환
        location = geolocator.geocode(address)
        
        if location:
            latitude = location.latitude
            longitude = location.longitude
            # 지도에 마커 추가
            folium.Marker([latitude, longitude], popup=f"{company_name}\n{address}").add_to(map)
        else:
            st.warning(f"주소를 찾을 수 없습니다: {address}")
    
    except Exception as e:
        st.error(f"주소 변환 중 오류 발생: {address} - {str(e)}")

# 스트림릿에서 지도를 표시
map_html = 'map.html'
map.save(map_html)

# 지도 HTML을 스트림릿에서 표시
st.components.v1.html(open(map_html, 'r').read(), height=500)
