import streamlit as st
from gensim.models import KeyedVectors as kv
import gensim.downloader
import numpy as np
from difflib import SequenceMatcher
import random as rdm
from commonWords import wordlist
from streamlit_js_eval import streamlit_js_eval
import os

def getClueList(model, words, weights) -> list:
    '''Returns an ordered list of words 
       that closely match the input words.'''
    vs = np.array([np.array(model[w]) for w in words])
    # cv: (weighted) center vector
    cs = np.array([vs[i] * weights[i] for i in range(len(weights))])
    cv = cs.sum(axis=0)
    cv = cv / np.linalg.norm(cv)
    return model.similar_by_vector(cv)

def matches(w1,w2) -> bool:
    '''Checks if two strings (kind of) match.'''
    return SequenceMatcher(None,str(w1),str(w2)).ratio()>0.8

def matchesNoneInList(w1,wl2) -> bool:
    '''Checks if a word is (kind of) in a wordlist.'''
    for w2 in wl2:
        if matches(w1,w2): return False
    return True

def getWord(secret, guess, wv) -> tuple[str,float]:
    '''Finds a word that is a good hint.'''
    oldwords=st.session_state.oldwords
    wordlist = [secret,guess]+oldwords
    weights = [1,1]+[-0.001 for i in range(len(oldwords))]
    cluelist = getClueList(wv,wordlist,weights)
    for word, fit in cluelist:
        if matchesNoneInList(word , oldwords +[secret]):
            return word, fit
    return None, None

def outText(outputText):
    '''Writes the output text.'''
    returnText.markdown(outputText)
    st.session_state.returnText=outputText

def history(histText):
    '''Adds text to the displayed history.'''
    st.session_state.historyText=histText + "\n\n" + st.session_state.historyText
    historyText.markdown(st.session_state.historyText)

def makeGuess(newGuess,secret,wv):
    '''Responds to a guess.'''
    outText("Please let me think about that for a moment.")
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
        newHint, fit =getWord(secret,newGuess,wv)
        if newHint==None:
            outText("Sorry, no clue found")
            return
        st.session_state.oldwords.append(newHint)
        sim=wv.similarity(secret,newGuess)*100
        outputtext= "\""+newGuess+ "\" is %.0f %% similar to the secret word. \n\n" %sim
        outputtext+="The secret word and the word \""\
                    +newGuess+\
                   "\" are both related to the word \""\
                   +newHint+"\" \n\n"
        if sim>=65:
            outputtext+="You are close. The first letter is: "\
                        +str(secret)[0]
        outText(outputtext)
        history(newGuess+ " "+ "%.0f %% " %sim + " -> " + newHint)

@st.cache_resource
def getwv():
    '''Gives cached wordvector or downloads it'''
    return gensim.downloader.load('glove-wiki-gigaword-100')

def initialization():
    '''Sets up session_state and thinks of secret word.'''
    os.environ['GENSIM_DATA_DIR'] = "."
    st.session_state.wv = getwv()
    wv = st.session_state.wv
    st.session_state.solved=False
    st.session_state.secret = ""
    while not st.session_state.secret in wv.key_to_index and\
            len(st.session_state.secret) <=6:
        st.session_state.secret=rdm.choice(wordlist).strip()
    st.session_state.returnText="init error"
    outText("Try to guess the secret word.") #overwrites init error
            # \n\n (always press enter before pressing guess.)")
    st.session_state.initialized = True
    st.session_state.oldwords = []
    st.session_state.historyText=""

def reload():
    streamlit_js_eval(js_expressions="parent.window.location.reload()")

def giveup():
    outText("The word was: "+st.session_state.secret)
    st.session_state.solved=True

# setup the page
st.title("Tims Word Game")
returnText = st.empty() # text box that talks to user
if 'initialized' not in st.session_state \
    or not st.session_state.initialized:
    initialization()
outText(st.session_state.returnText) # this often gets rerun

wv = st.session_state.wv # wordvector

# Create two columns for visualization
col1, col2 = st.columns([1,1], vertical_alignment="bottom") 
with col1:
    newGuess= str(st.text_input(label="",key="inputbox")).lower()
with col2:
    st.button(label="guess",on_click=makeGuess,\
              args=(newGuess,st.session_state.secret,wv),key="guess")
    
historyText= st.empty() # shows previous guesses
history("") # for all repeated calls
if st.session_state.solved:
    st.button(label="play again?",on_click=reload,\
              key="playagain")
else:
    st.button(label="give up?",on_click=giveup,key="giveup")

