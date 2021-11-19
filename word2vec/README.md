## Word2Vec implementation
### GloVe

### Skip-gram

### main.py
`main.py`는 fastapi를 통해 CBoW와 Skipgram을 웹에서 테스트할 수 있도록 api를 작성했습니다.

Test sentence : "We are about to study the idea of a computational process.  
Computational processes are abstract beings that inhabit computers.  
As they evolve, processes manipulate other abstract things called data.  
The evolution of a process is directed by a pattern of rules.  
called a program. People create programs to direct processes. In effect,  
we conjure the spirits of the computer with our spells."

Test center word : "People"

FastAPI에서 제공하는 Swagger UI를 통해 Skipgram을 테스트 한 결과입니다.
- Put string
![스크린샷 2021-11-17 오후 8 44 52](https://user-images.githubusercontent.com/50171632/142194669-725b5c38-c3e3-4a27-9a7e-1941b5b4c93c.png)

- Get center word
![스크린샷 2021-11-17 오후 8 45 08](https://user-images.githubusercontent.com/50171632/142194702-426ebf04-8811-4b1b-af25-30241aa499aa.png)
