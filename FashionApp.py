import streamlit as st
import pandas as pd
import plotly.express as px
import instaloader
from instaloader import Post
from instaloader.exceptions import BadResponseException, ConnectionException, LoginRequiredException, TooManyRequestsException
from streamlit_option_menu import option_menu

# Function to log in to Instagram
def login_instagram(username, password):
    L = instaloader.Instaloader()
    try:
        L.login(username, password)
        return L
    except (BadResponseException, ConnectionException, LoginRequiredException) as e:
        st.error(f"Instagram login failed: {str(e)}")
        return None

# Function to fetch Instagram post details with login
def fetch_instagram_post_details_with_login(L, link):
    try:
        post_shortcode = link.split('/')[-2]
        post = Post.from_shortcode(L.context, post_shortcode)

        # Initialize dictionary to hold the post data
        data = {
            "post_id": post.mediaid,
            "post_shortcode": post.shortcode,
            "post_date": post.date,
            "post_caption": post.caption,
            "post_likes": post.likes,
            "image_url": post.url,
            "video_url": post.video_url if post.is_video else None,
            "post_is_video": post.is_video,
            "post_hashtags": post.caption_hashtags,
            "post_mentions": post.caption_mentions,
            "comments": [comment.text for comment in post.get_comments()]
        }

        return data
    except (BadResponseException, ConnectionException, TooManyRequestsException) as e:
        st.error(f"Failed to fetch Instagram post details: {str(e)}")
        return {"error": str(e)}

# Function to determine if a post is trendy
def trendy_decision(likes_normalized, comments_normalized, sentiment_score, likes_th=0.8, comments_th=0.2, sentiment_th=0.1):
    is_trendy = (likes_normalized >= likes_th) & (sentiment_score >= sentiment_th) & (comments_normalized >= comments_th)
    return int(is_trendy)

# Custom CSS for dynamic background and centering text
st.markdown(
    """
    <style>
    .stApp {
        background: url("https://img.freepik.com/premium-photo/abstract-blurred-background-interior-clothing-store-shopping-mall_44943-543.jpg") no-repeat center center fixed;
        background-size: cover;
    }
    .centered-container {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100vh;
    }
    .centered-text {
        font-size: 100px;
        font-weight: bold;
        color: black;
    }
    </style>
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    selected = option_menu("Main Menu", ["Home", "Login", "Data", "EDA", "Post Preview", "Prediction"],
                           icons=['house', 'key', 'database', 'bar-chart', 'camera', 'cpu'],
                           menu_icon="cast", default_index=0)

# Load the dataset for the Data page
data_dataset_path = "C:/Users/Alaa/Desktop/Final_labeled_dataset.csv"
data_df = pd.read_csv(data_dataset_path)

# Load the dataset for the EDA page
eda_dataset_path = "C:/Users/Alaa/Desktop/FullDataset_Cleaned_Labelled_Normalized.csv"
eda_df = pd.read_csv(eda_dataset_path)

# Pages content
if selected == "Home":
    st.markdown(
        "<div class='centered-container'><div class='centered-text'>Fashion Trends</div></div>",
        unsafe_allow_html=True
    )

elif selected == "Data":
    st.title("Data Page")
    st.write("Display the contents of the dataset.")
    st.write(f"The dataset contains {data_df.shape[0]} rows and {data_df.shape[1]} columns.")

    # Text input to enter the number of rows to display
    num_rows_input = st.text_input("Enter the number of rows to display", value="5")
    if num_rows_input.isdigit():
        num_rows = int(num_rows_input)
        num_rows = min(num_rows, data_df.shape[0])  # Ensure num_rows does not exceed the number of rows in the dataset
        st.dataframe(data_df.head(num_rows))
    else:
        st.write("Please enter a valid number.")

elif selected == "EDA":
    st.title("EDA Page")
    st.write("Welcome to the EDA Page.")

    # Dropdown menu for selecting the graph
    graph_options = ["", "Distribution of Brands", "Sentiment Score Distribution", "Scatter Plot of Normalized Likes vs Sentiment Score",
                     "Box Plot of Normalized Likes by Brand", "Trend of Posts Over Time", "Sentiment Analysis by Brand", "Top Posts by Engagement"]
    selected_graph = st.selectbox("Select a graph to display", graph_options, index=0)

    if selected_graph == "Distribution of Brands":
        st.header("Distribution of Brands")
        brand_counts = eda_df['Brand'].value_counts()
        fig = px.bar(brand_counts, x=brand_counts.index, y=brand_counts.values, labels={'x': 'Brand', 'y': 'Count'})
        st.plotly_chart(fig)

    elif selected_graph == "Sentiment Score Distribution":
        st.header("Sentiment Score Distribution")
        fig = px.histogram(eda_df, x='sentiment_score', nbins=30)
        st.plotly_chart(fig)

    elif selected_graph == "Scatter Plot of Normalized Likes vs Sentiment Score":
        st.header("Scatter Plot of Normalized Likes vs Sentiment Score")
        fig = px.scatter(eda_df, x='Normalized_Likes', y='sentiment_score', color='Brand')
        st.plotly_chart(fig)

    elif selected_graph == "Box Plot of Normalized Likes by Brand":
        st.header("Box Plot of Normalized Likes by Brand")
        fig = px.box(eda_df, x='Brand', y='Normalized_Likes')
        st.plotly_chart(fig)

    elif selected_graph == "Trend of Posts Over Time":
        st.header("Trend of Posts Over Time")
        eda_df['post_date'] = pd.to_datetime(eda_df['post_date'])
        eda_df.set_index('post_date', inplace=True)
        post_trend = eda_df.resample('M').size()
        fig = px.line(post_trend, x=post_trend.index, y=post_trend.values, labels={'x': 'Date', 'y': 'Number of Posts'})
        st.plotly_chart(fig)

    elif selected_graph == "Sentiment Analysis by Brand":
        st.header("Sentiment Analysis by Brand")
        fig = px.box(eda_df, x='Brand', y='sentiment_score')
        st.plotly_chart(fig)

    elif selected_graph == "Top Posts by Engagement":
        st.header("Top Posts by Engagement")
        eda_df['engagement'] = eda_df['Normalized_Likes'] + eda_df['clean_comments'].apply(lambda x: len(x.split()))  # Assuming comments are space-separated
        top_posts = eda_df.nlargest(10, 'engagement')
        fig = px.bar(top_posts, x='post_id', y='engagement', hover_data=['post_caption'], labels={'post_id': 'Post ID', 'engagement': 'Engagement'})
        st.plotly_chart(fig)

elif selected == "Post Preview":
    st.title("Instagram Post Preview")

    if 'username' not in st.session_state or 'password' not in st.session_state:
        st.warning("Please login first")
        st.stop()

    username = st.session_state['username']
    password = st.session_state['password']

    instagram_link = st.text_input("Enter the Instagram post link:")

    # Display the preview of the Instagram post link
    if st.button("Fetch Post Details"):
        L = login_instagram(username, password)
        if L:
            data = fetch_instagram_post_details_with_login(L, instagram_link)

            if "error" in data:
                st.error(f"Failed to retrieve the post details. Error: {data['error']}")
            else:
                st.markdown("## Instagram Post Details")
                col1, col2 = st.columns([1, 2])
                with col1:
                    if data['post_is_video']:
                        st.video(data['video_url'])
                    else:
                        st.image(data['image_url'], caption='Instagram Post Image', use_column_width=True)
                with col2:
                    st.write(f"**Post ID:** {data['post_id']}")
                    st.write(f"**Post Shortcode:** {data['post_shortcode']}")
                    st.write(f"**Post Date:** {data['post_date']}")
                    st.write(f"**Caption:** {data['post_caption']}")
                    st.write(f"**Likes:** {data['post_likes']}")
                    st.write(f"**Is Video:** {'Yes' if data['post_is_video'] else 'No'}")
                    st.write(f"**Hashtags:** {', '.join(data['post_hashtags'])}")
                    st.write(f"**Mentions:** {', '.join(data['post_mentions'])}")

                st.markdown("## Comments")
                with st.expander("Show Comments"):
                    for comment in data['comments']:
                        st.write(comment)

        else:
            st.error("Failed to log in to Instagram. Please check your username and password.")

elif selected == "Prediction":
    st.title("Prediction Page")
    st.write("Determine if a post is trendy based on likes, comments, and sentiment scores.")

    if 'username' not in st.session_state or 'password' not in st.session_state:
        st.warning("Please login first")
        st.stop()

    username = st.session_state['username']
    password = st.session_state['password']

    instagram_link = st.text_input("Enter the Instagram post link:")

    # Input fields for thresholds
    likes_threshold = st.slider("Likes Threshold", 0.0, 1.0, 0.8)
    comments_threshold = st.slider("Comments Threshold", 0.0, 1.0, 0.2)
    sentiment_threshold = st.slider("Sentiment Threshold", 0.0, 1.0, 0.1)

    if st.button("Predict"):
        L = login_instagram(username, password)
        if L:
            data = fetch_instagram_post_details_with_login(L, instagram_link)

            if "error" in data:
                st.error(f"Failed to retrieve the post details. Error: {data['error']}")
            else:
                # Normalize the fetched data
                likes_normalized = data['post_likes'] / eda_df['Normalized_Likes'].max()
                comments_normalized = len(data['comments']) / eda_df['clean_comments'].apply(lambda x: len(x.split())).max()
                sentiment_score = 0.5  # Placeholder: Replace with actual sentiment analysis

                # Determine if the post is trendy
                is_trendy = trendy_decision(likes_normalized, comments_normalized, sentiment_score, likes_threshold, comments_threshold, sentiment_threshold)

                st.markdown("## Instagram Post Details")
                col1, col2 = st.columns([1, 2])
                with col1:
                    if data['post_is_video']:
                        st.video(data['video_url'])
                    else:
                        st.image(data['image_url'], caption='Instagram Post Image', use_column_width=True)
                with col2:
                    st.write(f"**Post ID:** {data['post_id']}")
                    st.write(f"**Post Shortcode:** {data['post_shortcode']}")
                    st.write(f"**Post Date:** {data['post_date']}")
                    st.write(f"**Caption:** {data['post_caption']}")
                    st.write(f"**Likes:** {data['post_likes']}")
                    st.write(f"**Is Video:** {'Yes' if data['post_is_video'] else 'No'}")
                    st.write(f"**Hashtags:** {', '.join(data['post_hashtags'])}")
                    st.write(f"**Mentions:** {', '.join(data['post_mentions'])}")
                    st.write(f"**Is Trendy:** {'Yes' if is_trendy else 'No'}")

                st.markdown("## Comments")
                with st.expander("Show Comments"):
                    for comment in data['comments']:
                        st.write(comment)

        else:
            st.error("Failed to log in to Instagram. Please check your username and password.")

elif selected == "Login":
    st.title("Instagram Login")

    # Input text boxes for the Instagram username and password
    username = st.text_input("Enter your Instagram username:")
    password = st.text_input("Enter your Instagram password:", type="password")

    if st.button("Login"):
        L = login_instagram(username, password)
        if L:
            st.session_state['username'] = username
            st.session_state['password'] = password
            st.success("Successfully logged in")
        else:
            st.error("Failed to log in. Please check your credentials.")
