"""
Fashion Trends Analysis App

This Streamlit app analyzes fashion trends based on Instagram posts from popular brands in the Middle East.
It involves logging into Instagram, selecting a brand, scraping post data, performing sentiment analysis on comments,
and determining the trendiness of the post.

Modules:
    - instaloader: For scraping Instagram post data.
    - pandas: For handling data in DataFrame format.
    - requests: For making HTTP requests.
    - re: For text processing using regular expressions.
    - PIL: For image processing.
    - groq: For sentiment analysis using the Groq API.
    - time: For measuring time taken for operations.
    - streamlit: For building the web application.
"""

#Import libraries
import streamlit as st
import instaloader
import pandas as pd
import requests
import re
from PIL import Image
from io import BytesIO
from getpass import getpass
from groq import Groq
import time
from instaloader import Post, BadResponseException, ConnectionException, TooManyRequestsException


# Constants
KEY = "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"
BACKGROUND_IMAGE_URL = "https://img.freepik.com/premium-photo/abstract-blurred-background-interior-clothing-store-shopping-mall_44943-543.jpg"

# Dictionary containing brand details and their stats
FASHION_BRANDS = {
    "HM": {
        "Platform": "Instagram",
        "URL": "https://www.instagram.com/hm/",
        "Region": "Middle East",
        "Stats": {
            "Followers": "38.4m",
            "Avg. likes": "19500", #19.5k
            "Avg. comments": "179.6"
        }
    },
    "Stradivarius": {
        "Platform": "Instagram",
        "URL": "https://www.instagram.com/stradivarius/",
        "Region": "Middle East",
        "Stats": {
            "Followers": "8.3m",
            "Uploads": "4k",
            "Avg. likes": "4400", #4.4k
            "Avg. comments": "19"
        }
    },
    "MANGO": {
        "Platform": "Instagram",
        "URL": "https://www.instagram.com/mango/",
        "Region": "Middle East",
        "Stats": {
            "Followers": "15.5m",
            "Avg. likes": "2100", #2.1k
            "Avg. comments": "49.8"
        }
    },
    "BERSHKA": {
        "Platform": "Instagram",
        "URL": "https://www.instagram.com/bershka/",
        "Region": "Middle East",
        "Stats": {
            "Followers": "10.9m",
            "Avg. likes": "12200", #12.2k
            "Avg. comments": "95.9"
        }
    },
    "PullandBear": {
        "Platform": "Instagram",
        "URL": "https://www.instagram.com/pullandbear/",
        "Region": "Middle East",
        "Stats": {
            "Followers": "7.8m",
            "Uploads": "5.9k",
            "Avg. likes": "7200", #7.2k
            "Avg. comments": "15.6"
        }
    },
    "Max Fashion Mena": {
        "Platform": "Instagram",
        "URL": "https://www.instagram.com/maxfashionmena/",
        "Region": "Middle East",
        "Stats": {
            "Followers": "2.7m",
            "Uploads": "8.2k",
            "Avg. likes": "248.5",
            "Avg. comments": "167.3"
        }
    },
    "Shein Arabia": {
        "Platform": "Instagram",
        "URL": "https://www.instagram.com/shein_ar/",
        "Region": "Middle East",
        "Stats": {
            "Followers": "5.5m",
            "Uploads": "10.1k",
            "Avg. likes": "536.8",
            "Avg. comments": "149.8"
        }
    },
    "Malameh Fashion": {
        "Platform": "Instagram",
        "URL": "https://www.instagram.com/mlameh_fashion_official/",
        "Region": "KSA",
        "Stats": {
            "Followers": "1.5m",
            "Uploads": "6.5k",
            "Avg. likes": "436.1",
            "Avg. comments": "25.9",
            "Avg. activity": "63.96%"
        }
    },
    "DeFacto": {
        "Platform": "Instagram",
        "URL": "https://www.instagram.com/defacto/",
        "Region": "Middle East",
        "Stats": {
            "Followers": "3.4m",
            "Uploads": "10.8k",
            "Avg. likes": "13700", #13.7k
            "Avg. comments": "915.1",
            "Avg. activity": "150.46%"
        }
    }
}


# Helper functions
def login_instagram(username, password):
    """
    Logs into Instagram using provided credentials.

    Args:
        username (str): Instagram username.
        password (str): Instagram password.

    Returns:
        instaloader.Instaloader: Logged in Instaloader instance or None if login fails.
    """
    L = instaloader.Instaloader()
    try:
        L.login(username, password)
        return L
    except Exception as e:
        st.error(f"Error during login: {e}")
        return None


def scrape_instagram_post(L, post_link):
    """
    Scrape Instagram post data using the provided username, password, and post link.
    Returns a DataFrame containing the post data.

    Args:
        L (instaloader.Instaloader): The Instaloader instance used for login.
        post_link (str): The link to the Instagram post.

    Returns:
        pandas.DataFrame: A DataFrame containing the post data.
    """

    try:
        start_time = time.time()

        # Extract shortcode from the post link
        shortcode = post_link.split("/")[-2]

        # Load the post using the shortcode
        post = Post.from_shortcode(L.context, shortcode)

        # Initialize dictionary to hold the post data
        data = {
            "post_id": post.mediaid,
            "post_shortcode": post.shortcode,
            "post_date": post.date,
            "post_caption": post.caption,
            "post_likes": post.likes,
            "image_url": post.url,
            "post_is_video": post.is_video,
            "post_hashtags": post.caption_hashtags,
            "post_mentions": post.caption_mentions,
            "video_url": post.url if post.is_video else None,
            "comments": []
        }

        # Get the total number of comments
        total_comments = post.comments
        post_likes = post.likes
        st.write(f"The post has {total_comments} comments and {post_likes} likes.")

        # Iterate over comments
        for i, comment in enumerate(post.get_comments(), start=1):
            if i % 5 == 0:
                st.write(f"Scraping comment {i} of {total_comments}")
            data["comments"].append(str(comment.text))

        # Convert the data to a DataFrame
        df = pd.DataFrame([data])

        end_time = time.time()
        time_taken = end_time - start_time
        st.write(f"Time taken to scrape the post: {time_taken:.2f} seconds")

        return df

    except (BadResponseException, ConnectionException, TooManyRequestsException) as e:
        st.error(f"Error fetching post data: {e}")
        return None


def clean_text(text):
    """
    Cleans the provided text by removing URLs, hashtags, mentions, and newlines/tabs.

    Args:
        text (str): Text to be cleaned.

    Returns:
        str: Cleaned text.
    """
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = text.replace('\n', ' ').replace('\t', ' ')
    return text


def select_model():
    """
    Prompts the user to select a sentiment analysis model from a list of options.

    Returns:
        str: The selected model identifier.
    """
    model_options = {
        "Gemma 7b": 'gemma-7b-it',
        "Mixtral 8x7b": 'mixtral-8x7b-32768',
        "LLaMA3 70b": 'llama3-70b-8192',
        "LLaMA3 8b": 'llama3-8b-8192'
    }
    selected_model = st.selectbox("Select the model to use for sentiment analysis", list(model_options.keys()), index=None)
    return model_options[selected_model] if selected_model else None


def analyze_sentiments_batch(client, comments, model, temperature=0.7):
    """
    Analyzes sentiments of the provided comments using the specified model.

    Args:
        client (groq.Client): Groq client for making API requests.
        comments (list of str): List of comments to be analyzed.
        model (str): The model identifier to use for sentiment analysis.
        temperature (float): Temperature parameter for the model.

    Returns:
        list of dict: List of sentiment analysis results.
    """
    
    prompt = "Analyze the sentiment of the following comments on fashion posts in context of fashion. Reply with 'positive', 'neutral', or 'negative' for each comment.\n\n"
    for i, comment in enumerate(comments, 1):
        prompt += f"{i}. {comment}\n"
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature
        )
    except Exception as e:
        st.error(f"Error in request: {e}")
        return None
    response_content = response.choices[0].message.content
    sentiments = []
    for line in response_content.split("\n"):
        if line.strip():
            if 'positive' in line.lower():
                sentiments.append('positive')
            elif 'neutral' in line.lower():
                sentiments.append('neutral')
            elif 'negative' in line.lower():
                sentiments.append('negative')
            else:
                sentiments.append('unknown')
    return sentiments


def count_sentiments(sentiments, sentiment_type):
    """
    Counts the number of occurrences of a specific sentiment type in a list of sentiments.

    Args:
        sentiments (list of str): List of sentiment strings.
        sentiment_type (str): The sentiment type to count (e.g., "positive", "negative", "neutral").

    Returns:
        int: The count of the specified sentiment type in the list. Returns 0 if the input is not a list.
    """
    if not isinstance(sentiments, list):
        return 0
    return sum(1 for sentiment in sentiments if sentiment.lower() == sentiment_type.lower())


def calculate_sentiment_scores(df):
    """
    Calculates the sentiment scores for a DataFrame containing sentiment analysis results.

    This function adds the following columns to the DataFrame:
        - "positive_count": Number of positive sentiments.
        - "negative_count": Number of negative sentiments.
        - "neutral_count": Number of neutral sentiments.
        - "total_sentiments": Total number of sentiments.
        - "sentiment_score": Normalized sentiment score calculated as 
          (positive_count - negative_count) / total_sentiments.

    Args:
        df (pandas.DataFrame): DataFrame containing a column "sentiments" with sentiment analysis results.
                               The "sentiments" column should contain string representations of lists.

    Returns:
        pandas.DataFrame: The DataFrame with the added sentiment analysis columns.
    """
    df["positive_count"] = df["sentiments"].apply(lambda x: count_sentiments(eval(x), "positive") if isinstance(x, str) else 0)
    df["negative_count"] = df["sentiments"].apply(lambda x: count_sentiments(eval(x), "negative") if isinstance(x, str) else 0)
    df["neutral_count"] = df["sentiments"].apply(lambda x: count_sentiments(eval(x), "neutral") if isinstance(x, str) else 0)
    df["total_sentiments"] = df["sentiments"].apply(lambda x: len(eval(x)) if isinstance(x, str) else 0)
    df["sentiment_score"] = (df["positive_count"] - df["negative_count"]) / df["total_sentiments"]
    return df


def normalize_likes_comments(df, avg_likes, avg_comments):
    """
    Normalizes the likes and comments in the DataFrame based on average likes and comments.

    This function adds the following columns to the DataFrame:
        - "likes_normalized": Normalized likes calculated as post_likes / avg_likes.
        - "comments_normalized": Normalized comments calculated as number of comments / avg_comments.

    Args:
        df (pandas.DataFrame): DataFrame containing the columns "post_likes" and "comments".
        avg_likes (float): Average number of likes to use for normalization.
        avg_comments (float): Average number of comments to use for normalization.

    Returns:
        pandas.DataFrame: The DataFrame with the added normalized likes and comments columns.
                          If avg_likes or avg_comments is 0, the function returns the original DataFrame.
    """
    
    if avg_likes == 0 or avg_comments == 0:
        return df
    df["likes_normalized"] = df["post_likes"] / avg_likes
    df["comments_normalized"] = df["comments"].apply(lambda x: len(x) if isinstance(x, list) else 0) / avg_comments
    return df


def trendy_decision(df, likes_th=0.8, comments_th=0.2, sentiment_th=0.1):
    """
    Determines whether a post is trendy based on normalized likes, comments, and sentiment score thresholds.

    This function adds a column "is_trendy" to the DataFrame, which indicates whether a post is considered trendy.
    A post is considered trendy if:
        - Normalized likes are greater than or equal to the likes threshold.
        - Normalized comments are greater than or equal to the comments threshold.
        - Sentiment score is greater than or equal to the sentiment threshold.

    Args:
        df (pandas.DataFrame): DataFrame containing the columns "likes_normalized", "comments_normalized", and "sentiment_score".
        likes_th (float, optional): Threshold for normalized likes. Default is 0.8.
        comments_th (float, optional): Threshold for normalized comments. Default is 0.2.
        sentiment_th (float, optional): Threshold for sentiment score. Default is 0.1.

    Returns:
        pandas.DataFrame: The DataFrame with the added "is_trendy" column indicating trendy posts.
    """
    df['is_trendy'] = ((df['likes_normalized'] >= likes_th) &
                      (df['comments_normalized'] >= comments_th) &
                      (df['sentiment_score'] >= sentiment_th))
    return df


# Streamlit app configuration
st.set_page_config(
    page_title="Fashion Trends Analysis",
    page_icon=":dress:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply background image
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("{BACKGROUND_IMAGE_URL}");
background-size: cover;
background-position: top;
background-attachment: scroll;
}}

[data-testid="stHeader"] {{
background: rgba(0, 0, 0, 0);
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# Streamlit app
if 'step' not in st.session_state:
    st.session_state.step = 0

def go_to_home():
    """
    Sets the current step to 0, effectively navigating to the home page.
    """
    st.session_state.step = 0

def go_to_next_step():
    """
    Increments the current step by 1, effectively navigating to the next step.
    """
    st.session_state.step += 1

def go_to_previous_step():
    """
    Decrements the current step by 1, effectively navigating to the previous step.
    """
    st.session_state.step -= 1

def reset_session_state():
    """
    Resets the session state by deleting all keys and setting the current step to 0.
    """
    for key in st.session_state.keys():
        del st.session_state[key]
    st.session_state.step = 0

# Landing page
if st.session_state.step == 0:
    st.title("Welcome to the Fashion Trends Analysis App")
    st.write("This app helps you analyze fashion trends based on Instagram posts from popular brands in the Middle East.")
    st.button("Start", on_click=go_to_next_step)

# Step 1: Login to Instagram
elif st.session_state.step == 1:
    st.title("Step 1: Login to Instagram")
    username = st.text_input("Enter your Instagram username", value="")
    password = st.text_input("Enter your Instagram password", type="password", value="")
    if st.button("Login"):
        if not username or not password:
            st.error("Please provide both username and password.")
        else:
            L = login_instagram(username, password)
            if L:
                st.success("Logged in successfully.")
                st.session_state.L = L

    # Button placement
    col1, _, col3 = st.columns([1, 8, 1])
    with col1:
        st.button("Previous", on_click=go_to_home)
    with col3:
        st.button("Next", on_click=go_to_next_step)

# Step 2: Select Brand
elif st.session_state.step == 2:
    st.title("Step 2: Select Brand")
    selected_brand = st.selectbox("Select a fashion brand", list(FASHION_BRANDS.keys()), index=None)
    if selected_brand:
        st.session_state.selected_brand = selected_brand  # Store selected brand in session state
        brand_info = FASHION_BRANDS[selected_brand]

        # Display brand information and stats in a boxed format with background color
        st.markdown(f"""
            <div style='background-color: #f0f0f0; padding: 20px; border-radius: 10px; box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);'>
                <h2>{selected_brand}</h2>
                <p><strong>Platform:</strong> {brand_info['Platform']}</p>
                <p><strong>URL:</strong> <a href='{brand_info['URL']}' target='_blank'>{brand_info['URL']}</a></p>
                <p><strong>Region:</strong> {brand_info['Region']}</p>
                <h3>Stats:</h3>
                <ul>
                    {"".join([f"<li><strong>{stat}:</strong> {value}</li>" for stat, value in brand_info['Stats'].items()])}
                </ul>
            </div>
        """, unsafe_allow_html=True)

    # Button placement
    col1, _, col3 = st.columns([1, 8, 1])
    with col1:
        st.button("Previous", on_click=go_to_previous_step)
    with col3:
        st.button("Next", on_click=go_to_next_step)


# Step 3: Enter Post Link
elif st.session_state.step == 3:
    st.title("Step 3: Enter Post Link")
    post_link = st.text_input("Enter the Instagram post link", value="")
    if st.button("Fetch Post"):
        if not post_link:
            st.error("Please provide a post link.")
        else:
            df = scrape_instagram_post(st.session_state.L, post_link)
            if df is not None:
                st.session_state.df = df

                # Display fetched post details in a structured manner
                st.markdown("### Fetched Post Details")
                col1, col2 = st.columns([1, 2])
                with col1:
                    if df['post_is_video'].iloc[0]:
                        st.video(df['video_url'].iloc[0])
                    else:
                        st.image(df['image_url'].iloc[0], caption='Instagram Post Image', use_column_width=True)
                with col2:
                    st.write(f"**Post ID:** {df['post_id'].iloc[0]}")
                    st.write(f"**Post Shortcode:** {df['post_shortcode'].iloc[0]}")
                    st.write(f"**Post Date:** {df['post_date'].iloc[0]}")
                    st.write(f"**Caption:** {df['post_caption'].iloc[0]}")
                    st.write(f"**Likes:** {df['post_likes'].iloc[0]}")
                    st.write(f"**Is Video:** {'Yes' if df['post_is_video'].iloc[0] else 'No'}")
                    st.write(f"**Hashtags:** {' '.join(df['post_hashtags'].iloc[0])}")
                    st.write(f"**Mentions:** {' '.join(df['post_mentions'].iloc[0])}")

                # Display the DataFrame
                st.markdown("### Scraped Post Data")
                st.write(df)

    # Button placement
    col1, _, col3 = st.columns([1, 8, 1])
    with col1:
        st.button("Previous", on_click=go_to_previous_step)
    with col3:
        st.button("Next", on_click=go_to_next_step)


# Step 4: Perform Sentiment Analysis
elif st.session_state.step == 4:
    st.title("Step 4: Perform Sentiment Analysis")
    if "df" in st.session_state and not st.session_state.df.empty:
        st.write("Post Data:", st.session_state.df[["post_caption", "post_likes", "post_hashtags", "post_mentions"]])
        model = select_model()
        if model:
            comments = st.session_state.df["comments"].iloc[0]
            clean_comments = [clean_text(comment) for comment in comments]
            if st.button("Analyze Sentiments"):
                client = Groq(api_key=KEY)
                sentiments = analyze_sentiments_batch(client, clean_comments, model)
                if sentiments:
                    st.session_state.df["sentiments"] = str(sentiments)
                    st.write("Sentiments Analysis:", sentiments)

    # Button placement
    col1, _, col3 = st.columns([1, 8, 1])
    with col1:
        st.button("Previous", on_click=go_to_previous_step)
    with col3:
        st.button("Next", on_click=go_to_next_step)

# Step 5: Sentiment Analysis Results
elif st.session_state.step == 5:
    st.title("Step 5: Sentiment Analysis Results")
    if "df" in st.session_state and not st.session_state.df.empty:
        df = st.session_state.df
        df = calculate_sentiment_scores(df)
        st.write("Sentiment Scores:", df[["positive_count", "negative_count", "neutral_count", "sentiment_score"]])
        avg_likes = float(FASHION_BRANDS[st.session_state.selected_brand]["Stats"].get("Avg. likes", 0))
        avg_comments = float(FASHION_BRANDS[st.session_state.selected_brand]["Stats"].get("Avg. comments", 0))
        df = normalize_likes_comments(df, avg_likes, avg_comments)
        st.write("Normalized Likes and Comments:", df[["likes_normalized", "comments_normalized"]])
        df = trendy_decision(df)
        st.session_state.result_df = df

    # Button placement
    col1, _, col3 = st.columns([1, 8, 1])
    with col1:
        st.button("Previous", on_click=go_to_previous_step)
    with col3:
        st.button("Next", on_click=go_to_next_step)



# Step 6: Final Decision on Trendiness
elif st.session_state.step == 6:
    st.title("Step 6: Trendy Decision")
    if "result_df" in st.session_state and not st.session_state.result_df.empty:
        df = st.session_state.result_df
        is_trendy = df["is_trendy"].iloc[0]
        
        # Display trendy decision in a boxed format with background color
        if is_trendy:
            st.markdown("""
                <div style='background-color: #c8e6c9; padding: 10px; border-radius: 5px;'>
                    <p style='color: green;'>The analyzed post is trendy!</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div style='background-color: #ffcdd2; padding: 10px; border-radius: 5px;'>
                    <p style='color: red;'>The analyzed post is not trendy.</p>
                </div>
            """, unsafe_allow_html=True)

    # Button placement
    col1, _, _ = st.columns([1, 8, 1])
    with col1:
        st.button("Previous", on_click=go_to_previous_step)



