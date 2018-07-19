import tweepy
from Scrypers.Twitter import credentials

# Counter to keep track of results
hashtag_count = 0

# Hashtag/Pound symbol in unicode format
hashtag_symbol = "%23"

# Prompt the user for input. Returns the results in lowercase format.
hashtag_keyword = input("What is the hashtag you'd like to search for? No need to include the '#' key: ").lower()

# Combine the unicode hashtag symbol
hashtag_search = hashtag_symbol + hashtag_keyword

# Define the search function
def run_search():
    hashtag_results = tweepy.Cursor(credentials.api.search, q=hashtag_search, result_type='recent', include_rts = False,
                                    exclude_replies = True).items(3000)
    return hashtag_results

# Get hashtag count
for tweet in run_search():
    hashtag_count += 1

if hashtag_count == 0:
    print('There are ' + str(hashtag_count) + ' tweets containing the hashtag: \'' + hashtag_keyword + '\'')
elif hashtag_count == 1:
    print('There is ' + str(hashtag_count) + ' tweet containing the hashtag: \'' + hashtag_keyword + '\'')
else:
    print('There are ' + str(hashtag_count) + ' tweets containing the hashtag: \'' + hashtag_keyword + '\'')

# We're going to store information in a text file
# Open the file. If it doesn't exist, python creates it
savefile = open("hipokryci.csv", 'a', encoding='utf-8')

# Print the results to the console
# Write the results to the text file
for tweet in run_search():
    print("Tweeted by: @" + tweet.user.screen_name)
    tweet = "Tweeted by: @" + tweet.user.screen_name + "\n" + str(tweet.text) + "\n\n"
    tweet = tweet.replace("b\'", "\'")
    tweet = tweet.replace("b\"", "\"")
    savefile.write(tweet)

# Close the text file.
savefile.close()
