import pandas as pd
import requests
from keymanager import KeyManager as km
import mythril
import hashlib

### Python 3.9 Required
class Oppenheimer:

    

    def __init__(self):
        pass

    def getBytecode(address):
        url="https://eth-mainnet.g.alchemy.com/v2/{apikey}".format(apikey=km().Easy_Key("alchemy-key2"))
        payload={
            "id": 1,
            "jsonrpc": "2.0",
            "method": "eth_getCode"
        }
        payload["params"]=[address, "latest"]
        headers={
            "accept": "application/json",
            "content-type": "application/json"
        }
        return requests.post(url, json=payload, headers=headers).json()["result"]

    def getChar(self,opcode):
        if "df" not in globals().keys():
            global df
            df=pd.read_csv("opcode_to_char.csv")
        if opcode in df['operation'].values:
            df=df[['char_index','characters','operation']]
            return df[df['operation']==opcode]['characters'].values[0]
        else:
            indexes=df['char_index'].tolist()
            last_char_index=max(indexes)
            character=chr(last_char_index)
            while character.isprintable()==False or last_char_index in indexes:
                last_char_index+=1
                character=chr(last_char_index)
            newdf=pd.DataFrame(data={'char_index':[last_char_index],"characters":[character],"operation":opcode})
            df=pd.concat([df,newdf])
            df=df[['char_index','characters','operation']]
            df.to_csv("opcode_to_char.csv")
            return character

    def processOpcode(self,bytecode):
        opcodes = mythril.disassembler.asm.disassemble(bytecode)
        hashablelist=[]
        string_compare=""
        for op in opcodes:
            opcode=op['opcode']
            hashablelist.append(opcode)
            string_compare=string_compare+Oppenheimer().getChar(opcode)
        return string_compare
    
    def processOpcodewithHash(self,bytecode):
        opcodes = mythril.disassembler.asm.disassemble(bytecode)
        hashablelist=[]
        string_compare=""
        for op in opcodes:
            opcode=op['opcode']
            hashablelist.append(opcode)
            string_compare=string_compare+Oppenheimer().getChar(opcode)
        op_hash=hashlib.sha1(bytes("".join(hashablelist),encoding='UTF-8')).hexdigest()
        return op_hash,string_compare

    ### Example Usage
    # op_hash,string_compare=processOpcode(getBytecode("0xb7B4B6D077fc59E6387C3c4ff9a9a6BE031d1dfE"))
    # op_hash2,string_compare2=processOpcode(getBytecode("0x1D795dc2c31645ce3efD7c51253851C6015f2818"))
    # print(textdistance.ratcliff_obershelp.normalized_similarity(string_compare,string_compare2))



