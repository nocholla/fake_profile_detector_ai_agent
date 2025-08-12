import streamlit as st
from src.run import FakeProfileDetector

def main():
    st.title("🚩 Fake Profile Detector")
    st.write("Enter profile details to detect potential fake profiles.")
    
    with st.form("profile_form"):
        age = st.number_input("Age", min_value=18, max_value=100, value=21)
        country = st.text_input("Country", value="Ghana")
        about_me = st.text_area("About Me", value="Hi I'm looking for serious love, not games.")
        relationship_goals = st.selectbox("Relationship Goals", ["Long-term", "Short-term", "Marriage", "Casual", "unknown"])
        subscribed = st.checkbox("Subscribed", value=False)
        submitted = st.form_submit_button("Detect Profile")
        
        if submitted:
            profile = {
                'age': age,
                'country': country,
                'aboutMe': about_me,
                'relationshipGoals': relationship_goals,
                'subscribed': subscribed
            }
            detector = FakeProfileDetector()
            try:
                detector.load_models()
                result = detector.predict(profile)
                st.write(f"**Prediction**: {'Fake' if result['is_fake'] else 'Real'}")
                st.write(f"**Confidence**: {result['confidence']:.2f}")
                st.write("**Reasons**:")
                for reason in result['reasons']:
                    st.write(f"- {reason}")
                st.write(f"**Similar Profile Indices**: {result['similar_profiles']}")
            except Exception as e:
                st.error(f"Error: {e}")

if __name__ == "__main__":
    main()