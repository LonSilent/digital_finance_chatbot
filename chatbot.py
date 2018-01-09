import os
import time
import re
from slackclient import SlackClient
import json
from sklearn.feature_extraction.text import TfidfVectorizer
import pysparnn.cluster_index as ci
import jieba
from retrieval import build_index, build_index_new
from collections import defaultdict

jieba.set_dictionary('./dict/dict_zh_small.txt')

jieba.load_userdict('./dict/user_dict_zh.txt')
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
        print(event)
        if event["type"] == "message" and not "subtype" in event:
            user_id, message = parse_direct_mention(event["text"])
            if user_id == starterbot_id and event["channel"][0] == 'C':
                return message, event["channel"], event["user"]
            else:
                return event["text"], event["channel"], event["user"]
    return None, None, None

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

# def parse_result(search_data, tv, all_cp, key):
#     search_data = [search_data]

#     search_data_cut = [' '.join(jieba.cut(x)) for x in search_data]
#     search_features_vec = tv.transform(search_data_cut)

#     result = all_cp[key].search(search_features_vec.getrow(0), k=1, return_distance=False)
#     return result[0][0].replace(' ','')

def handle_command(command, channel, tv, cp, user_name, user_data, user_count):
    """
        Executes bot command if the command is known
    """
    user_count[user_name] += 1
    key_set = ['信用卡', '金融卡', '存款與繳款', '外匯', '貸款', '財富管理與保險', '其他']

    # Default response is help text for the user
    default_response = "我是智能聊天客服機器人 :slightly_smiling_face: 使用方法： `我想問 [option]` 。 \noption有下列七大項： `信用卡` , `金融卡` , `存款與繳款` , `外匯` , `貸款` , `財富管理與保險` , `其他` 。 \nexample: `我想問 信用卡`"

    help_response = "使用方法：我想問 [option]。 \noption有下列七大項： `信用卡` , `金融卡` , `存款與繳款` , `外匯` , `貸款` , `財富管理與保險` , `其他` 。\nexample: `我想問 信用卡` \n ----------------------------\nreset機器人: `clear` \n使用方法說明: `help`"

    # Finds and executes the given command, filling in response
    # response = None
    # This is where you start to implement more commands!
    # if command.startswith(EXAMPLE_COMMAND):
    #     response = "Sure...write some more code then I can do that!"
    response = None
    if command.startswith('我想問 '):
        key = command.split(' ')[-1]
        if key in key_set:
            response = 'OK，收到。你想問 `{}` 的問題484 :smiley: 來吧都來！'.format(key)
            user_data[user_name] = key
        else:
            response = '不好意思，你問的東西不在我能處理的範圍喔 :joy: 嘗試用 `help` 看看怎樣跟我互動吧'
    elif command.startswith('help'):
        response = help_response
    elif command.startswith('clear'):
        response = 'clear for debug'
        if user_name in user_data:
            del user_data[user_name]
        user_count[user_name] = 0
    elif user_count[user_name] == 1:
        response = default_response
    elif user_name in user_data:
        key = user_data[user_name]
        response = parse_result(command, tv, cp[key])

    else:
        response = '我不知道你要問什麼耶 :joy: 嘗試用 `help` 看看怎樣跟我互動吧'

    # Sends the response back to the channel
    slack_client.api_call(
        "chat.postMessage",
        channel=channel,
        text=response or default_response
    )

if __name__ == "__main__":
    user_data = {}
    user_count = defaultdict(int)
    # tv, cp = build_index('qa.json')
    tv, all_cp = build_index_new('qa_final.json')

    print("Finish Load Data...")

    if slack_client.rtm_connect(with_team_state=False):
        print("ChatBot connected and running!")
        # Read bot's user ID by calling Web API method `auth.test`
        starterbot_id = slack_client.api_call("auth.test")["user_id"]
        while True:
            command, channel, user_name = parse_bot_commands(slack_client.rtm_read())
            # print("command:", command, "channel:", channel)
            if command:
                handle_command(command, channel, tv, all_cp, user_name, user_data, user_count)
            time.sleep(RTM_READ_DELAY)
    else:
        print("Connection failed. Exception traceback printed above.")