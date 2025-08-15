import streamlit as st
import re
from joblib import load

model = load("news_detector.joblib")

st.markdown("""
    <style>
    .block-container {
        padding-top: 1.2 rem;
    }
    </style> """, unsafe_allow_html=True)

def clean_text(text):
    text = text.lower()
    text =re.sub(r'\[.*?\]', '', text)
    text =re.sub(r"\W", " ", text)
    text =re.sub(r'https?://\S+|www\.\S+', '', text)
    text =re.sub(r'<.*?>+', '', text)
    text =re.sub(r'\w*\d\w*', '', text)
    return text

st.set_page_config(page_title="News Detector",page_icon="📰")
st.title("📰 Fake News Detector")

user_input = st.text_area("Enter News Article:",value=st.session_state.get("text",""), height=300)
col1,col2=st.columns(2)
with col2:
    butt=st.button("Check")
if butt:
        if not user_input.strip():
            st.error("⚠️ Please enter some news text first.")
        clean = clean_text(user_input)
        prediction = model.predict([clean])[0]
        result = "✅ Real News" if prediction == 1 else "❌ Fake News"
        st.subheader(result)
        proba = model.predict_proba([clean])[0]
        st.metric("Detection Probability:", f"{proba[1]*100:.2f}%")

st.sidebar.title("⚙️ Model Settings")
if st.sidebar.button("Demo Real News"):
    st.session_state["text"]="MOSCOW (Reuters) - Russiaâ€™s former ambassador to " \
    "Washington, Sergei Kislyak, said on Saturday his conversations with former " \
    "White House national security adviser Michael Flynn had been transparent and focused" \
    " on matters of U.S.-Russia cooperation. Kislyak ended his tenure in Washington in July " \
    "but remains a key figure in ongoing U.S. investigations into Moscowâ€™s alleged meddling in" \
    " the 2016 presidential election. Flynn was forced to resign in February after it became known" \
    " that he had failed to disclose the content of conversations he had with Kislyak and misled U.S." \
    " Vice-President Mike Pence about their meetings. â€œWe only spoke about the most simple things .." \
    ". but the communication was completely correct, calm, absolutely transparent. In any case, there " \
    "were no secrets on our side,â€ Kislyak said during a panel discussion on Russian television. " \
    "â€œThere are a number of issues which are important for cooperation between Russia and the United " \
    "States - most of all, terrorism. And that was one of the things we discussed.â€ "
    st.rerun()

if st.sidebar.button("Demo Fake News"):
    st.session_state['text']="Federal health officials told the AP they have not received any reports of Ebola cases " \
    "at the Nevada event. A screenshot of a supposed post from the Centers for Disease Control and Prevention confirming " \
    "such cases was fabricated. And there is no record of a national emergency being declared. The claims emerged after summe" \
    "r storm left muddy roads flooded, stranding tens of thousands of partygoers; event organizers let traffic flow out of the main " \
    "road Monday afternoon. “So it was announced earlier that Burning Man was declared a national emergency because it was flooded," \
    " and so they sent in FEMA,” a woman claims in a TikTok video shared on Instagram, suggesting the development was suspicious. The" \
    " AP found no record, including on federal websites and in White House announcements, of a national emergency declaration and FEMA " \
    "confirmed that it was not involved in the situation. “No FEMA personnel or assets have been deployed to the Burning Man festival" \
    " and there are no requests from local or state authorities for our assistance,” FEMA spokesperson Jeremy Edwards said in an email." \
    " The TikTok video, like other posts, goes on to relay baseless rumors of reported cases of Ebola, whose occasional outbreaks in " \
    "humans primarily occur in Africa, at the festival. Some posts also shared an image made to appear that the CDC confirmed the" \
    " supposed outbreak on X, the platform formerly known as Twitter. The purported X post from the agency reads, “Ebola outbreak " \
    "confirmed at Black Rock City, NV. It is recommended that all Burning Man attendees remain in their dwellings until further notice." \
    " Current State of Emergency in progress.” But the CDC’s X account published no such post. “CDC has not received any reports of Ebola" \
    " at the Burning Man Festival and has not issued any warnings or had any requests for assistance from the state and local health " \
    "departments either,” agency spokesperson Scott Pauley said in an email. Reverse image searches further show that a graphic about" \
    " Ebola used in the fictitious CDC post was published by the agency in 2016, but elements of it were changed. For example, " \
    "the original graphic asks, “Recently in West Africa?” But the version used in the made-up X post asks, “Recently in Nevada?” " \
    "Referencing more online rumors, Pauley also noted the CDC had not received reports of mpox, formerly known as monkeypox, " \
    "or Marburg, a rare but severe hemorrhagic fever, in relation to Burning Man. A representative for the Burning Man Project " \
    "organization also refuted the online claims. “Quite simply, the online rumors of transmissible illnesses in Black Rock City" \
    " are unfounded and untrue,” Dominique Debucquoy-Dodley said in an email. The festival had been closed to vehicles after more " \
    "than a half-inch (1.3 centimeters) of rain fell Sept. 1, causing flooding and foot-deep mud, as the AP reported. The annual " \
    "gathering, which launched on a San Francisco beach in 1986, attracts nearly 80,000 artists, musicians and activists for a mix " \
    "of wilderness camping and avant-garde performances"

    st.rerun()
