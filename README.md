# Experiments with embedding for trees

Numerical experiments for the model [embeddings-for-trees](https://github.com/JetBrains-Research/embeddings-for-trees)

Experiment results are stored as follows (test set is the same among all datasets):
  * [Experiments on random 10% of full java-med](https://wandb.ai/strange_attractor/tree-lstm-java-med-10per1-finals?workspace=user-strange_attractor)
	  Global methods:
	  |    Method			|	BLEU	|	meteor	|	rouge-1	|	rouge-2	|	rouge-l	|
	  |:-------------------:|:----------|:---------:|:---------:|:---------:|:---------:|
	  |SGD					|0.2777		|0.2647		|0.3839		|0.1597		|0.3940		|
	  |Adam					|0.2474		|0.2443		|0.3556		|0.1416		|0.3651		|
	  |RAdam				|0.3108		|0.2784		|0.3879		|0.1781		|0.4010		|
	  |Lamb					|0.3158		|0.2853		|0.3980		|0.1809		|0.4095		|
	  |LaSGD				|0.2451		|0.2475		|0.3689		|0.1424		|0.3792		|
	  |LaRAdam				|**0.3647**	|**0.3151**	|**0.4310**	|**0.2080**	|**0.4428**	|
	  |Upside SGD-LaRAdam, %|31.3		|19.1		|12.3		|30.3		|12.4		|
	  
	  Local methods:
	  |    Method				|	BLEU	|	meteor	|	rouge-1	|	rouge-2	|	rouge-l	|
	  |:-----------------------:|:----------|:---------:|:---------:|:---------:|:---------:|
	  |Adadelta					|0.3705		|0.3191		|0.4317		|0.2126		|0.4430		|
	  |BB						|**0.3863**	|**0.3268**	|**0.4414**	|**0.2172**	|**0.4531**	|
	  |LaRAdam					|0.3843		|0.3260		|0.4400		|0.2171		|0.4511		|
	  |SVRG						|0.3663		|0.3172		|0.4306		|0.2107		|0.4419		|
	  |SWA						|0.3851		|0.3263		|0.4407		|0.2173		|0.4523		|
	  |Upside LaRAdam(global)-BB|5.9		|3.7		|2.4		|4.4		|2.3		|
  
  * [Quick check of SGD-RAdam rangeing on another 10% random subset](https://wandb.ai/strange_attractor/tree-lstm-java-med-10per2-finals?workspace=user-strange_attractor)
  * [Experiments on full java-med](https://wandb.ai/strange_attractor/tree-lstm-java-med-asts-finals?workspace=user-strange_attractor)
 	  Global methods:
 	  |    Method			|	BLEU	|	meteor	|	rouge-1	|	rouge-2	|	rouge-l	|
	  |:-------------------:|:----------|:---------:|:---------:|:---------:|:---------:|
	  |SGD					|0.3822		|0.3310		|0.4479		|0.2213		|0.4592		|
	  |LaRAdam3				|0.3644		|0.3246		|0.4446		|0.2169		|0.4573		|
	  |LaRAdam7				|0.3758		|0.3299		|0.4503		|0.2209		|0.4634		|
	  |RAdam(normal)		|---		|---		|---		|---		|---		|
	  |Upside, %			|---		|---		|---		|---		|---		|
	  
	  Local metods:
	  |    Method			|	BLEU	|	meteor	|	rouge-1	|	rouge-2	|	rouge-l	|
	  |:-------------------:|:----------|:---------:|:---------:|:---------:|:---------:|
	  |SWA					|0.3890		|0.3371		|0.4550		|0.2273		|0.4660		|
	  |Upside SGD-SWA, %    |1.8		|1.8		|1.6		|2.7		|1.5		|
	
 	
  * [Check of LaRAdam, SGD 10-->25% score improvement](https://wandb.ai/strange_attractor/tree-lstm-java-med-25per-finals?workspace=user-strange_attractor)
  	Global methods (1: columns format is metric/upside to 10% train set):
 	  |    Method			|	BLEU	|	meteor	|	rouge-1	|	rouge-2	|	rouge-l	|
	  |:-------------------:|:----------|:---------:|:---------:|:---------:|:---------:|
	  |SGD					|---		|---		|---		|---		|---		|
	  |LaRAdam				|---		|---		|---		|---		|---		|
  
Please find model outputs in both tensor and string form [here](https://www.dropbox.com/sh/u0dn37mebrwk99t/AAAYJuKMwb1M_MhTfUnDkQTia?dl=0)  (zip files with corresponding method name).

