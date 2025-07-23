# Ad Analysis Dashboard - Enhanced and Fixed Version
import openai
from openai import OpenAI
import streamlit as st
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from textblob import TextBlob
import re
import ssl
import os
from sklearn.feature_extraction.text import CountVectorizer

# NLTK SETUP


@st.cache_resource
def setup_nltk():
    try:
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context

        required_resources = {
            'punkt': 'tokenizers/punkt',
            'stopwords': 'corpora/stopwords',
            'wordnet': 'corpora/wordnet'
        }

        for resource, path in required_resources.items():
            try:
                nltk.data.find(path)
            except LookupError:
                nltk.download(resource, quiet=True, force=True)
        return True
    except Exception as e:
        st.error(f"Failed to initialize NLTK: {str(e)}")
        return False

# DATA LOADING


@st.cache_data
def load_data():
    if not os.path.exists('vintamine.xlsx'):
        st.warning("vintamine.xlsx not found. Please upload the file.")
        uploaded_file = st.file_uploader(
            "Upload vintamine.xlsx", type=["xlsx"])
        if uploaded_file:
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        else:
            return None
    else:
        df = pd.read_excel('vintamine.xlsx', engine='openpyxl')

    column_map = {
        'text_columns': ['ad_text', 'text', 'content', 'description', 'caption', 'ad_content'],
        'company_columns': ['company_name', 'company', 'brand', 'advertiser', 'client'],
        'type_columns': ['add_type', 'ad_type', 'type', 'media_type', 'creative_type'],
        'cta_columns': ['call_to_action', 'cta', 'action', 'button_text']
    }

    renamed_cols = {}
    for col_type, possibilities in column_map.items():
        for possible_col in possibilities:
            if possible_col in df.columns:
                standard_name = {
                    'text_columns': 'ad_text',
                    'company_columns': 'company_name',
                    'type_columns': 'add_type',
                    'cta_columns': 'call_to_action'
                }[col_type]
                renamed_cols[possible_col] = standard_name
                break

    df = df.rename(columns=renamed_cols)

    required_cols = ['company_name', 'add_type', 'call_to_action']
    for col in required_cols:
        if col not in df.columns:
            st.error(f"Missing required column: {col}")
            return None

    if 'ad_text' not in df.columns:
        text_cols = [col for col in df.columns if df[col].dtype ==
                     'object' and col not in required_cols]
        if text_cols:
            df['ad_text'] = df[text_cols[0]]
        else:
            st.error("No text column found for ad_text")
            return None

    df['ad_text'] = df['ad_text'].fillna('').astype(str)
    df['call_to_action'] = df['call_to_action'].fillna('Unknown')
    df['add_type'] = df['add_type'].fillna('Unknown')

    if 'total_active_time' in df.columns:
        df['duration_numeric'] = pd.to_numeric(
            df['total_active_time'], errors='coerce')

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

    return df

# TEXT UTILITIES


def preprocess_text(text):
    if not isinstance(text, str) or not text.strip():
        return []
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    stop_words = set(stopwords.words('english'))
    return [word for word in tokens if word not in stop_words and len(word) > 2 and word.isalpha()]


def get_sentiment(text):
    blob = TextBlob(str(text))
    return blob.sentiment.polarity, blob.sentiment.subjectivity

# GPT-4 AD COPY GENERATOR


def generate_ad_copy(keywords, tone="friendly", product_name="Ventamin"):
    prompt = f"""
    You are a skilled ad copywriter. Based on the following:
    - Product: {product_name}
    - Keywords: {', '.join(keywords)}
    - Tone: {tone}
    
    Generate:
    1. A compelling ad headline (max 10 words)
    2. A short ad body (1-2 sentences)
    3. A suitable call-to-action (CTA)

    Format:
    Headline: ...
    Body: ...
    CTA: ...
    """

    client = OpenAI(api_key=openai.api_key)
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating ad copy: {str(e)}"

# MAIN DASHBOARD


def main():
    if not setup_nltk():
        st.stop()

    # Initialize OpenAI API with multiple fallback options
    openai.api_key = None
    try:
        # Try Streamlit secrets first
        openai.api_key = st.secrets["openai"]["api_key"]
    except Exception as e:
        st.sidebar.warning(f"Failed to load secrets: {str(e)}")

    if not openai.api_key:
        # Try environment variable
        openai.api_key = os.getenv("OPENAI_API_KEY")

    if not openai.api_key:
        # Try manual input (for testing)
        with st.sidebar.expander("🔑 OpenAI API Key Setup"):
            api_key_input = st.text_input(
                "Enter OpenAI API Key (optional)", type="password")
            if api_key_input:
                openai.api_key = api_key_input
                st.success("API key set for this session")

    gpt_enabled = bool(openai.api_key)
    if not gpt_enabled:
        st.sidebar.warning("GPT-4 features disabled - no API key found")

    st.title("📊 Competitor Ad Analysis Dashboard")
    st.markdown("Analyze and visualize competitor ads with NLP insights")

    df = load_data()
    if df is None:
        st.stop()

    # FILTERS SECTION
    st.sidebar.header("🔍 Filters")
    company_filter = st.sidebar.selectbox(
        "Select Company", ["All"] + sorted(df['company_name'].unique()))
    ad_type_filter = st.sidebar.selectbox(
        "Select Ad Type", ["All"] + sorted(df['add_type'].unique()))
    cta_filter = st.sidebar.selectbox(
        "Select CTA", ["All"] + sorted(df['call_to_action'].unique()))

    if 'date' in df.columns:
        date_range = st.sidebar.date_input(
            "Date Range", [df['date'].min().date(), df['date'].max().date()])

    # Apply filters
    filtered_df = df.copy()
    if company_filter != "All":
        filtered_df = filtered_df[filtered_df['company_name']
                                  == company_filter]
    if ad_type_filter != "All":
        filtered_df = filtered_df[filtered_df['add_type'] == ad_type_filter]
    if cta_filter != "All":
        filtered_df = filtered_df[filtered_df['call_to_action'] == cta_filter]
    if 'date' in df.columns and len(date_range) == 2:
        filtered_df = filtered_df[
            (filtered_df['date'].dt.date >= date_range[0]) &
            (filtered_df['date'].dt.date <= date_range[1])
        ]

    if filtered_df.empty:
        st.warning("No ads found for the selected filters.")
        st.stop()

    # OVERVIEW SECTION
    st.header("📈 Overview Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Ads", len(filtered_df))
    col2.metric("Unique Companies", filtered_df['company_name'].nunique())
    col3.metric("Ad Types", filtered_df['add_type'].nunique())

    st.subheader("🔎 Search Ads by Keyword")
    keyword = st.text_input("Enter keyword")
    if keyword:
        results = filtered_df[filtered_df['ad_text'].str.contains(
            keyword, case=False)]
        st.write(f"Found {len(results)} results")
        st.dataframe(results[['company_name', 'ad_text']])

    # TEXT ANALYSIS SECTION
    st.header("📝 Text Analysis")
    all_text = ' '.join(filtered_df['ad_text'].astype(str))
    tokens = preprocess_text(all_text)

    if tokens:
        # Word Cloud
        word_freq = Counter(tokens)
        wc = WordCloud(width=800, height=400,
                       background_color='white').generate_from_frequencies(word_freq)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

        # Top Keywords
        st.subheader("Top Keywords")
        st.dataframe(pd.DataFrame(word_freq.most_common(
            20), columns=['Keyword', 'Count']))

    # Bigram Analysis
    st.subheader("Bigram Analysis")
    vectorizer = CountVectorizer(ngram_range=(2, 2), stop_words='english')
    X = vectorizer.fit_transform(filtered_df['ad_text'])
    bigram_df = pd.DataFrame(
        X.sum(axis=0).tolist()[0],
        index=vectorizer.get_feature_names_out(),
        columns=['Count']
    ).reset_index()
    bigram_df.columns = ['Bigram', 'Count']
    st.dataframe(bigram_df.sort_values('Count', ascending=False).head(15))

    # VISUALIZATIONS SECTION
    st.header("📊 Visualizations")

    # Ad Type Distribution
    ad_type_counts = filtered_df['add_type'].value_counts().reset_index()
    ad_type_counts.columns = ['Ad Type', 'Count']
    st.plotly_chart(px.pie(ad_type_counts, names='Ad Type',
                    values='Count', title="Ad Type Distribution"))

    # Top CTAs
    cta_counts = filtered_df['call_to_action'].value_counts().reset_index()
    cta_counts.columns = ['CTA', 'Count']
    st.plotly_chart(px.bar(cta_counts.head(10), x='CTA',
                    y='Count', title="Top CTAs"))

    # Monthly Ad Volume (if date available)
    if 'date' in filtered_df.columns:
        timeline = filtered_df.groupby(filtered_df['date'].dt.to_period(
            'M')).size().reset_index(name='Ad Count')
        timeline['date'] = timeline['date'].astype(str)
        st.plotly_chart(px.line(timeline, x='date',
                        y='Ad Count', title='Monthly Ad Volume'))

    # CTA Usage by Company
    st.subheader("CTA Usage by Company")
    pivot_cta = pd.crosstab(
        filtered_df['company_name'], filtered_df['call_to_action'])
    st.dataframe(pivot_cta.style.background_gradient(cmap="Blues"))

    # SENTIMENT ANALYSIS SECTION
    st.subheader("Sentiment Analysis")
    filtered_df[['polarity', 'subjectivity']] = filtered_df['ad_text'].apply(
        get_sentiment).apply(pd.Series)
    st.caption(
        "Polarity: -1 (negative) to 1 (positive), Subjectivity: 0 (objective) to 1 (subjective)")
    st.plotly_chart(px.scatter(filtered_df, x='polarity', y='subjectivity',
                               color='company_name', hover_data=['ad_text'],
                               title='Polarity vs Subjectivity'))

    # Sentiment Summary by Company
    st.subheader("Sentiment Summary by Company")
    sent_summary = filtered_df.groupby('company_name')[
        ['polarity', 'subjectivity']].mean().reset_index()
    st.plotly_chart(px.bar(sent_summary, x='company_name',
                    y='polarity', title='Average Polarity by Company'))

    # AD LENGTH ANALYSIS
    st.subheader("Ad Length Analysis")
    filtered_df['word_count'] = filtered_df['ad_text'].apply(
        lambda x: len(x.split()))
    filtered_df['length_bucket'] = pd.cut(
        filtered_df['word_count'],
        bins=[0, 10, 25, 50, 100, 500],
        labels=['Very Short', 'Short', 'Medium', 'Long', 'Very Long']
    )
    st.plotly_chart(px.histogram(filtered_df, x='length_bucket',
                    title='Ad Length Distribution'))

    # Extreme Sentiment Ads
    st.subheader("🔥 Extreme Sentiment Ads")
    extreme_ads = filtered_df[(filtered_df['polarity'] > 0.8) | (
        filtered_df['polarity'] < -0.8)]
    st.dataframe(extreme_ads[['company_name', 'ad_text', 'polarity']])

    # DOWNLOAD SECTION
    st.header("📥 Download")
    st.download_button(
        "Download Filtered Data as CSV",
        data=filtered_df.to_csv(index=False).encode('utf-8'),
        file_name='filtered_ad_data.csv',
        mime='text/csv'
    )

    # GPT-4 AD COPY GENERATOR SECTION
    if gpt_enabled:
        st.header("💡 GPT-4 Ad Copy Generator")
        with st.expander("Generate ad copy based on keywords and tone"):
            keywords_input = st.text_input(
                "Enter keywords (comma-separated)", "immunity, clean energy, daily supplement")
            tone_input = st.selectbox(
                "Select tone", ["Friendly", "Professional", "Inspiring", "Bold", "Playful"])
            product_name = st.text_input("Product name", "Ventamin")

            if st.button("Generate Ad Copy"):
                with st.spinner("Generating with GPT-4..."):
                    try:
                        keywords = [k.strip()
                                    for k in keywords_input.split(",") if k.strip()]
                        result = generate_ad_copy(
                            keywords, tone_input.lower(), product_name)
                        st.markdown("### ✍️ Generated Ad Copy")
                        st.code(result, language='markdown')
                    except openai.AuthenticationError:
                        st.error(
                            "Invalid OpenAI API key. Please check your key and try again.")
                    except openai.RateLimitError:
                        st.error("Rate limit exceeded. Please try again later.")
                    except Exception as e:
                        st.error(f"Error generating ad copy: {e}")
    else:
        st.warning("GPT-4 features disabled - no API key found")


if __name__ == '__main__':
    main()
