from web3 import Web3
### use web3==5.31.1 and python 3.9.0
import polars as pl
import mythril
import math


def getChar(opdf,opcode):
    search=opdf.filter(pl.col("operation")==opcode)
    if search.height>0:
        return opdf,search.select('characters').item()
    else:
        max_in_idx=opdf['char_index'].max()+1
        character=chr(max_in_idx)
        while character.isprintable()==False:
            character=chr(max_in_idx)
            max_in_idx+=1
        newdf=pl.DataFrame(data={'char_index':[max_in_idx],"characters":[character],"operation":opcode})
        opdf=pl.concat([opdf,newdf])
        opdf.write_parquet("opcode_to_char.parquet")
        return opdf,character

def processOpcode(opdf,bytecode):
    opcodes = mythril.disassembler.asm.disassemble(bytecode)
    oplist=[]
    for op in opcodes:
        opcode=op['opcode']
        opdf,character=getChar(opdf,opcode)
        oplist.append(character)
    return opdf,"".join(oplist)

# df=pl.read_parquet("opcode_to_char.parquet")
# print(getChar(df,"PUSH"))
#op=Oppenheimer()

def main():
    web3 = Web3(Web3.HTTPProvider("https://reth.shield3.com/rpc"))
    df=pl.read_parquet('data_with_fam.parquet')
    opdf=pl.read_parquet("opcode_to_char.parquet")
    addresses=df['address'].to_list()
    families=df['family'].to_list()
    counter=0
    addrs=[]
    bytecodes=[]
    opcodes=[]
    opcodes_merged=[]

    for fam in families:
        if fam==None:
            continue
        addrs.append(addresses[counter])
        bytecodes_smol=[]
        opcodes_smol=[]
        opx=[]
        unique_codes=[]
        for addr in fam:
            print('processing '+addr)
            bytecode=web3.eth.get_code(Web3.toChecksumAddress(addr)).hex()
            
            if bytecode in unique_codes:
                continue
            unique_codes.append(bytecode)
            opdf,ops=processOpcode(opdf,bytecode)
            if ops in opx:
                continue
            
            opx.append(ops)
            bytecodes_smol.append(bytecode)
            opcodes_smol.append(ops)
        bytecodes.append(bytecodes_smol)
        opcodes.append(opcodes_smol)
        opcodes_merged.append("".join(opcodes_smol))
        counter+=1
        if counter/10==math.trunc(counter/10):
            newdf=pl.DataFrame({'bytecodes':bytecodes,'opcodes_chars':opcodes,'merged_opcodes':opcodes_merged})
            finaldf=pl.concat([df,newdf],how='horizontal')
            finaldf.write_parquet('datafiles/data_with_code{}.parquet'.format(counter))
            print('chunk saved')
    newdf=pl.DataFrame({'bytecodes':bytecodes,'opcodes_chars':opcodes,'merged_opcodes':opcodes_merged})
    finaldf=pl.concat([df,newdf],how='horizontal')
    finaldf.write_parquet('datafiles/data_with_code{}.parquet'.format(counter))
    print('process complete')


main()