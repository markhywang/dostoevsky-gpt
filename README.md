# Dostoevsky GPT

Fyodor Mikhailovich Dostoevsky (1821 - 1881) was a Russian author and novelist, and is arguably one of the greatest novelists in the history of literature. Notably, towards the end of his life, Dostoevsky was in the process of writing a novel titled Nietoshka Nezvanova. However, unfortunately, he passed away before he is able to finish the book.

Therefore, for fun of course, I will build my own Transformer GPT model (that is, a decoder-only transformer with character-wise tokenization) inside this notebook that will learn off of all of what is written on the unfinished novel Nietoshka Nezvanova and then generate future text. This future text is what the model thinks should be how the story continues (even though, of course, this is probably not what Dostoevsky himself would have wrote had he finished the book...). Thus, essentially, in the end we can generate and infer our own fully-finished versions of Nietoshka Nezvanova!

Note that, due to computation limitations and the fact that I am not willing to spend months training an LLM, the transformer model is rather small. Thus, the generation texts will not be as accurate or sensible compared to a more sophisticated GPT.

Here are some things a curious learner with an absolute beast of a NVIDIA GPU can do to (but not limited to) make this model better:
1. Enlarge the model by increasing hyperparameters such as
    - number of attention heads,
    - number of unfolding layers,
    - max number of tokens the model can process at once
    - more training epochs,
    - etc.
2. Use a more robust tokenization technique such as Byte-Pair Encoding (BPE), WordPiece, etc.
3. First pre-train the transformer on some of Dostoevsky's other works to learn more about his writing style and any linguistic/contextual complexities
   to better adapt the model to his writings. Afterwards, fine-tune the model specifically on Nietoshka Nezvanova to have the model generate text  specific to that novel
   - In fact, a text file of The Brothers Karamazov, Dostoevsky's longest work, is attached under /data for you to try and pre-train.

Credits:
- The text file for Netochka Nezvanova is directly taken from: https://www.thetedkarchive.com/library/fyodor-dostoevsky-netochka-nezvanova-an-incomplete-novel
- The text file for The Brothers Karamazov is directly taken from: https://www.gutenberg.org/ebooks/28054
- This transformer architecture was inspired by [Andrej Karpathy](https://github.com/karpathy). The link to the GitHub repository can be found here: https://github.com/karpathy/ng-video-lecture

Also, the amazing paper titled "Attention Is All You Need" which first proposed the possibility of Transformer models can be found here: https://arxiv.org/abs/1706.03762

![alt text](https://github.com/markhywang/Dostoevsky-GPT/blob/main/dostoevsky-image.jpg)
