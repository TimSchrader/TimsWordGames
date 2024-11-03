import streamlit as st
from gensim.models import KeyedVectors as kv
import numpy as np
from difflib import SequenceMatcher
import random as rdm
from commonWords import wordlist
from streamlit_js_eval import streamlit_js_eval

def weighted_between(model, ws, gs): # ws: words gs: weights
    vs = np.array([np.array(model[w]) for w in ws])
    cs = np.array([vs[i] * gs[i] for i in range(len(gs))])# cv: (weighted) center vector
    cv = cs.sum(axis=0)
    cv = cv / np.linalg.norm(cv)
    return model.similar_by_vector(cv)

def matches(w1,w2):
    return SequenceMatcher(None,str(w1),str(w2)).ratio()>0.8

def matchesNoneInList(w1,wl2):
    for w2 in wl2:
        if matches(w1,w2): return False
    return True

def getWord(secret, guess, wv):
    oldwords=st.session_state.oldwords
    wordlist = [secret,guess]+oldwords
    weights = [1,1]+[-0.1 for i in range(len(oldwords))]
    cluelist = weighted_between(wv,wordlist,weights)
    for word, fit in cluelist:
        if matchesNoneInList(word , oldwords +[secret]):
            return word
    return None

def outText(outputText):
    returnText.markdown(outputText)
    st.session_state.returnText=outputText

def makeGuess(newGuess,secret,wv):
    outText("Let me think about that.")
    if len(newGuess)<=1: 
        outText("The guess \""+newGuess+"\" is to short.")
        return
    elif secret not in wv.key_to_index: 
        outText("secret not known, my bad.")
        return
    elif newGuess not in wv.key_to_index: 
        outText("Sorry, I don't know the word "+newGuess)
        return
    elif newGuess==secret: 
        outText("You solved it! Hurray!")
        return
    else:
        st.session_state.oldwords.append(newGuess)
        newHint=getWord(secret,newGuess,wv)
        if newHint==None:
            outText("Sorry, no clue found")
            return
        st.session_state.oldwords.append(newHint)
        outText( \
                "The secret word and the word \n\""\
                +newGuess+\
                "\"\n are both related to the word\n\""\
                +newHint+"\"")

def initialization(wv):
    st.session_state.solved=False
    st.session_state.secret = ""
    while not st.session_state.secret in wv.key_to_index:
        st.session_state.secret=rdm.choice(wordlist).strip()
    st.session_state.returnText="init error"
    outText("Try to guess the secret word.  \
            (always press enter before pressing guess.)") #overwrites init error
    st.session_state.initialized = True
    st.session_state.oldwords = []
def reload():
    streamlit_js_eval(js_expressions="parent.window.location.reload()")

def giveup():
    outText("The word was: "+st.session_state.secret)
    st.session_state.solved=True

#from https://huggingface.co/fse/glove-wiki-gigaword-300/tree/main
wv = kv.load("glove-wiki-gigaword-300.model", mmap='r')

st.title("Tims Word Game")
returnText = st.empty() # text box that talks to user
if 'initialized' not in st.session_state \
    or not st.session_state.initialized:
    initialization(wv)
outText(st.session_state.returnText) # this often gets rerun

# Create two columns; adjust the ratio ?
col1, col2 = st.columns([1,1],\
                        vertical_alignment="bottom") 
with col1:
    newGuess= str(st.text_input(label="",key="inputbox")).lower()
with col2:
    st.button(label="guess",on_click=makeGuess,\
              args=(newGuess,st.session_state.secret,wv),key="guess")
if st.session_state.solved:
    st.button(label="play again?",on_click=reload,\
              key="playagain")
else:
    st.button(label="give up?",on_click=giveup,key="giveup")

