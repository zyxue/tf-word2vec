The `word2vec_basic.py` from
[tensorflow repo](https://github.com/tensorflow/tensorflow/blob/r1.7/tensorflow/examples/tutorials/word2vec/word2vec_basic.py)
not modularized and is hard to read (a copy is included), so I

* refactored it;
* made certain variables (e.g. `embedding_size`, `skip_window`, `batch_size`) tunable via command lnie
* ensure it produces the same graph and results.

**Graph**

![graph](https://raw.githubusercontent.com/zyxue/tf-word2vec/master/graph.png)


**Output**

```
Average loss at step  92000 :  4.6741576662063595
Average loss at step  94000 :  4.728303222894668
Average loss at step  96000 :  4.681813739180565
Average loss at step  98000 :  4.5938096576333045
Average loss at step  100000 :  4.711780580878258
2018-04-03 17:10:36,478|INFO|Nearest to UNK: dasyprocta, agouti, cebus, michelob, circ, ursus, operatorname, constituci,
2018-04-03 17:10:36,484|INFO|Nearest to the: its, their, agouti, his, constituci, ursus, a, this,
2018-04-03 17:10:36,488|INFO|Nearest to of: ursus, in, including, pulau, eight, gigantopithecus, operatorname, for,
2018-04-03 17:10:36,493|INFO|Nearest to and: or, but, circ, dasyprocta, michelob, while, peacocks, constituci,
2018-04-03 17:10:36,497|INFO|Nearest to one: six, two, four, seven, eight, dasyprocta, five, operatorname,
2018-04-03 17:10:36,501|INFO|Nearest to in: during, within, at, on, from, of, through, with,
2018-04-03 17:10:36,506|INFO|Nearest to a: the, any, gigantopithecus, koruna, another, kapoor, alcatraz, pulau,
2018-04-03 17:10:36,510|INFO|Nearest to to: thaler, through, must, would, will, taxonomy, circ, ursus,
2018-04-03 17:10:36,514|INFO|Nearest to zero: eight, five, four, seven, nine, six, three, dasyprocta,
2018-04-03 17:10:36,519|INFO|Nearest to nine: eight, seven, six, five, zero, four, three, dasyprocta,
2018-04-03 17:10:36,523|INFO|Nearest to two: three, four, five, six, seven, eight, one, operatorname,
2018-04-03 17:10:36,527|INFO|Nearest to is: was, has, are, operatorname, became, aediles, gigantopithecus, dasyprocta,
2018-04-03 17:10:36,532|INFO|Nearest to as: dasyprocta, constituci, gigantopithecus, wct, roshan, masonry, operatorname, backslash,
2018-04-03 17:10:36,536|INFO|Nearest to eight: seven, nine, six, five, four, zero, three, dasyprocta,
2018-04-03 17:10:36,540|INFO|Nearest to for: in, including, thuringiensis, constituci, operatorname, of, with, ursus,
2018-04-03 17:10:36,545|INFO|Nearest to s: his, thaler, hbox, dasyprocta, ursus, constituci, mountaineer, four,
```
