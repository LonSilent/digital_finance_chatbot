import os
import time
import re
from slackclient import SlackClient
import json
from sklearn.feature_extraction.text import TfidfVectorizer
import pysparnn.cluster_index as ci
import jieba
from retrieval import build_index

jieba.set_dictionary('../zh-dict/dict_zh_small.txt')

jieba.load_userdict('../zh-dict/acg.txt')
jieba.load_userdict('../zh-dict/ec_item_zh.txt')
jieba.load_userdict('../zh-dict/user_dict_zh.txt')
jieba.load_userdict('./dict/finance_dict.txt')

# instantiate Slack client
slack_client = SlackClient(os.environ.get('SLACK_BOT_TOKEN'))
# starterbot's user ID in Slack: value is assigned after the bot starts up
starterbot_id = None

# constants
RTM_READ_DELAY = 1 # 1 second delay between reading from RTM
EXAMPLE_COMMAND = "我想問"
MENTION_REGEX = "^<@(|[WU].+)>(.*)"

def parse_bot_commands(slack_events):
    """
        Parses a list of events coming from the Slack RTM API to find bot commands.
        If a bot command is found, this function returns a tuple of command and channel.
        If its not found, then this function returns None, None.
    """
    for event in slack_events:
        if event["type"] == "message" and not "subtype" in event:
            user_id, message = parse_direct_mention(event["text"])
            if user_id == starterbot_id:
                return message, event["channel"]
    return None, None

def parse_direct_mention(message_text):
    """
        Finds a direct mention (a mention that is at the beginning) in message text
        and returns the user ID which was mentioned. If there is no direct mention, returns None
    """
    matches = re.search(MENTION_REGEX, message_text)
    # the first group contains the username, the second group contains the remaining message
    return (matches.group(1), matches.group(2).strip()) if matches else (None, None)

def parse_result(search_data, tv, cp):
    search_data = [search_data]

    search_data_cut = [' '.join(jieba.cut(x)) for x in search_data]
    search_features_vec = tv.transform(search_data_cut)

    result = cp.search(search_features_vec.getrow(0), k=1, return_distance=False)
    return result[0][0].replace(' ','')

def handle_command(command, channel, tv, cp):
    """
        Executes bot command if the command is known
    """
    # Default response is help text for the user
    # default_response = "Not sure what you mean. Try *{}*.".format(EXAMPLE_COMMAND)

    # Finds and executes the given command, filling in response
    # response = None
    # This is where you start to implement more commands!
    # if command.startswith(EXAMPLE_COMMAND):
    #     response = "Sure...write some more code then I can do that!"

    response = parse_result(command, tv, cp)

    # Sends the response back to the channel
    slack_client.api_call(
        "chat.postMessage",
        channel=channel,
        text=response or default_response
    )

if __name__ == "__main__":
    tv, cp = build_index('qa.json')

    print("Finish Load Data...")

    if slack_client.rtm_connect(with_team_state=False):
        print("Starter Bot connected and running!")
        # Read bot's user ID by calling Web API method `auth.test`
        starterbot_id = slack_client.api_call("auth.test")["user_id"]
        while True:
            command, channel = parse_bot_commands(slack_client.rtm_read())
            print("command:", command, "channel:", channel)
            if command:
                handle_command(command, channel, tv, cp)
            time.sleep(RTM_READ_DELAY)
    else:
        print("Connection failed. Exception traceback printed above.")