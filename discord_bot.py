import discord #discord用関数
from transformers import T5Tokenizer, AutoModelForCausalLM #発言生成用
import time#時間制御用
import re #名前を変更する際に正規表現を使用


TOKEN = 'TOKEN'  # 自身のDiscord用のTOKENを貼り付け
CHANNELID = 0000000000000000000 # botを開放したいチャンネルIDを貼り付け(int)

#discord bot使用のためのおまじない
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

#モデル読み込み
tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-gpt2-medium")#入出力文をトークンに変換する際の基準
tokenizer.do_lower_case = True#計算遼を軽減
model = AutoModelForCausalLM.from_pretrained("/")#モデルの入っているディレクトリのパスを入れる

#GPU使用するなら記載する


#漢字の正規表現
chinese_chr = re.compile('[\u2E80-\u2FDF\u3005-\u3007\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF\U00020000-\U0002EBEF]+')

#安倍晋三の発言生成用関数
def generate_talk(othersay: str, num=1) -> None:
    
    #入力テキストをトークン化

    input_text = '<s>'+othersay+'[SEP]'

    input_ids = tokenizer.encode(input_text, return_tensors='pt', add_special_tokens=False)
    
    #。で終了する文章ができるまで生成
    while True:
        out = model.generate(
            input_ids,
            do_sample=True,
            top_k=100,
            top_p=0.95,
            max_length=150,
            temperature=0.95,
            num_return_sequences=num,
            bad_words_ids=[[tokenizer.bos_token_id], [tokenizer.sep_token_id], [tokenizer.unk_token_id]],
            repetition_penalty=1.2,
        )
        #生成した文章トークンをstrに変換して不要部削除
        output=str(tokenizer.batch_decode(out))
        output = output.replace('<s>', '')
        output = output.split('[SEP]')[1]
        output = output.replace('</s>', '').strip()

        #生成した文に。が含まれていたら生成を終了
        if '。' in output:
            break
    
    #生成した文章を2文にして。を付け直す
    outlist=output.split('。')
    output=outlist[0]+"。"
    
    #生成した文章を関数の戻り値に
    return output



#discord botの起動時に行う処理
@client.event
async def on_ready():
    
    #書き込むチャンネルを指定
    channel=client.get_channel(CHANNELID)

    #起動メッセージと起動確認
    await channel.send('おはようございます。ご質問にお答えいたします。')
    print('起動確認')

#discord botにおいて書き込み用チャンネルに書き込まれた際に行う処理
@client.event
async def on_message(message):
    
    #書き込み主が自分なら無視
    if message.author == client.user:
        return

    #書き込みが自分以外なら会話生成
    else:
        #書き込むチャンネルを取得
        channel=client.get_channel(CHANNELID)
        
        #メッセージ内容とメッセージ主を取得
        input=message.content
        auth=message.author.name
        firstname=[]
        output=generate_talk(input)

        #メッセージ内の名前部を変換

        #メッセージ内に名前が出てこないとき
        if re.search(r'委員|議員|さん',output ) == None:
            n=len(output)
            time.sleep(n*0.2)
            await channel.send(output)
            print('メッセージ送信成功')
        
        #メッセージ内名前が出てきそうなとき
        else:
            #委員・議員の始まる位置を特定
            marker=re.search(r'委員|議員|さん',output ).start()

            #上記の位置から遡って連続する漢字(苗字部分)を抽出
            for i in range(1, len(output)):
                fnchr=output[marker-i]
                
                #抽出部が漢字でなければ終了
                if chinese_chr.fullmatch(fnchr) == None:
                    break
                
                #漢字なら苗字リストに入れる
                else:
                    firstname.append(fnchr)
                    firstname.reverse()
            
            if len(firstname) == 0:
                n=len(output)
                #文字数で送信時間制御
                time.sleep(n*0.2)
                await channel.send(output)
                print('メッセージ送信成功')

            else:
                fn=''.join(firstname)

                #上記の苗字を書き込み主の名前に変換してメッセージ送信
                newoutpt=output.replace(fn, auth)
                n=len(newoutpt)
                #文字数で送信時間制御
                time.sleep(n*0.2)
                await channel.send(newoutpt)
                print('置換メッセージ送信成功')

#稼働させる
client.run(TOKEN)