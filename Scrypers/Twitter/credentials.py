import tweepy

# Enter the corresponding information from your Twitter application management:
CONSUMER_KEY = '' # Keep the quotes, replace this with your consumer key
CONSUMER_SECRET = '' # Keep the quotes, replace this with your consumer secret key
ACCESS_TOKEN = '' # Keep the quotes, replace this with your access token
ACCESS_SECRET = '' # Keep the quotes, replace this with your access secret key

# Configure our access information for reaching Twitter
auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)

# Access Twitter!
api = tweepy.API(auth, wait_on_rate_limit = True)
