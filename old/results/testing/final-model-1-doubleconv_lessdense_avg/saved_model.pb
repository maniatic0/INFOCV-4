Ί΄
Ώ£
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
Ύ
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.3.02unknown8Συ

conv2d_99/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_nameconv2d_99/kernel
}
$conv2d_99/kernel/Read/ReadVariableOpReadVariableOpconv2d_99/kernel*&
_output_shapes
:@*
dtype0
t
conv2d_99/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_99/bias
m
"conv2d_99/bias/Read/ReadVariableOpReadVariableOpconv2d_99/bias*
_output_shapes
:@*
dtype0

conv2d_100/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*"
shared_nameconv2d_100/kernel

%conv2d_100/kernel/Read/ReadVariableOpReadVariableOpconv2d_100/kernel*&
_output_shapes
:@@*
dtype0
v
conv2d_100/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_nameconv2d_100/bias
o
#conv2d_100/bias/Read/ReadVariableOpReadVariableOpconv2d_100/bias*
_output_shapes
:@*
dtype0
~
dense_198/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*!
shared_namedense_198/kernel
w
$dense_198/kernel/Read/ReadVariableOpReadVariableOpdense_198/kernel* 
_output_shapes
:
*
dtype0
u
dense_198/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_198/bias
n
"dense_198/bias/Read/ReadVariableOpReadVariableOpdense_198/bias*
_output_shapes	
:*
dtype0
}
dense_199/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*!
shared_namedense_199/kernel
v
$dense_199/kernel/Read/ReadVariableOpReadVariableOpdense_199/kernel*
_output_shapes
:	@*
dtype0
t
dense_199/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_199/bias
m
"dense_199/bias/Read/ReadVariableOpReadVariableOpdense_199/bias*
_output_shapes
:@*
dtype0
|
dense_200/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@
*!
shared_namedense_200/kernel
u
$dense_200/kernel/Read/ReadVariableOpReadVariableOpdense_200/kernel*
_output_shapes

:@
*
dtype0
t
dense_200/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_200/bias
m
"dense_200/bias/Read/ReadVariableOpReadVariableOpdense_200/bias*
_output_shapes
:
*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h
VariableVarHandleOp*
_output_shapes
: *
dtype0	*
shape:*
shared_name
Variable
a
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes
:*
dtype0	
l

Variable_1VarHandleOp*
_output_shapes
: *
dtype0	*
shape:*
shared_name
Variable_1
e
Variable_1/Read/ReadVariableOpReadVariableOp
Variable_1*
_output_shapes
:*
dtype0	
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0

Adam/conv2d_99/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/conv2d_99/kernel/m

+Adam/conv2d_99/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_99/kernel/m*&
_output_shapes
:@*
dtype0

Adam/conv2d_99/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_99/bias/m
{
)Adam/conv2d_99/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_99/bias/m*
_output_shapes
:@*
dtype0

Adam/conv2d_100/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*)
shared_nameAdam/conv2d_100/kernel/m

,Adam/conv2d_100/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_100/kernel/m*&
_output_shapes
:@@*
dtype0

Adam/conv2d_100/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdam/conv2d_100/bias/m
}
*Adam/conv2d_100/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_100/bias/m*
_output_shapes
:@*
dtype0

Adam/dense_198/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*(
shared_nameAdam/dense_198/kernel/m

+Adam/dense_198/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_198/kernel/m* 
_output_shapes
:
*
dtype0

Adam/dense_198/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_198/bias/m
|
)Adam/dense_198/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_198/bias/m*
_output_shapes	
:*
dtype0

Adam/dense_199/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*(
shared_nameAdam/dense_199/kernel/m

+Adam/dense_199/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_199/kernel/m*
_output_shapes
:	@*
dtype0

Adam/dense_199/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/dense_199/bias/m
{
)Adam/dense_199/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_199/bias/m*
_output_shapes
:@*
dtype0

Adam/dense_200/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@
*(
shared_nameAdam/dense_200/kernel/m

+Adam/dense_200/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_200/kernel/m*
_output_shapes

:@
*
dtype0

Adam/dense_200/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/dense_200/bias/m
{
)Adam/dense_200/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_200/bias/m*
_output_shapes
:
*
dtype0

Adam/conv2d_99/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/conv2d_99/kernel/v

+Adam/conv2d_99/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_99/kernel/v*&
_output_shapes
:@*
dtype0

Adam/conv2d_99/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_99/bias/v
{
)Adam/conv2d_99/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_99/bias/v*
_output_shapes
:@*
dtype0

Adam/conv2d_100/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*)
shared_nameAdam/conv2d_100/kernel/v

,Adam/conv2d_100/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_100/kernel/v*&
_output_shapes
:@@*
dtype0

Adam/conv2d_100/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdam/conv2d_100/bias/v
}
*Adam/conv2d_100/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_100/bias/v*
_output_shapes
:@*
dtype0

Adam/dense_198/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*(
shared_nameAdam/dense_198/kernel/v

+Adam/dense_198/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_198/kernel/v* 
_output_shapes
:
*
dtype0

Adam/dense_198/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_198/bias/v
|
)Adam/dense_198/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_198/bias/v*
_output_shapes	
:*
dtype0

Adam/dense_199/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*(
shared_nameAdam/dense_199/kernel/v

+Adam/dense_199/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_199/kernel/v*
_output_shapes
:	@*
dtype0

Adam/dense_199/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/dense_199/bias/v
{
)Adam/dense_199/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_199/bias/v*
_output_shapes
:@*
dtype0

Adam/dense_200/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@
*(
shared_nameAdam/dense_200/kernel/v

+Adam/dense_200/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_200/kernel/v*
_output_shapes

:@
*
dtype0

Adam/dense_200/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/dense_200/bias/v
{
)Adam/dense_200/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_200/bias/v*
_output_shapes
:
*
dtype0

NoOpNoOp
?N
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ιM
valueίMBάM BΥM

layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer_with_weights-1
layer-4
layer-5
layer-6
layer-7
	layer_with_weights-2
	layer-8

layer_with_weights-3

layer-9
layer_with_weights-4
layer-10
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
y
layer-0
layer-1
layer-2
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
R
regularization_losses
 	variables
!trainable_variables
"	keras_api
R
#regularization_losses
$	variables
%trainable_variables
&	keras_api
h

'kernel
(bias
)regularization_losses
*	variables
+trainable_variables
,	keras_api
R
-regularization_losses
.	variables
/trainable_variables
0	keras_api
R
1regularization_losses
2	variables
3trainable_variables
4	keras_api
R
5regularization_losses
6	variables
7trainable_variables
8	keras_api
h

9kernel
:bias
;regularization_losses
<	variables
=trainable_variables
>	keras_api
h

?kernel
@bias
Aregularization_losses
B	variables
Ctrainable_variables
D	keras_api
h

Ekernel
Fbias
Gregularization_losses
H	variables
Itrainable_variables
J	keras_api

Kiter

Lbeta_1

Mbeta_2
	Ndecay
Olearning_ratemΆm·'mΈ(mΉ9mΊ:m»?mΌ@m½EmΎFmΏvΐvΑ'vΒ(vΓ9vΔ:vΕ?vΖ@vΗEvΘFvΙ
 
F
0
1
'2
(3
94
:5
?6
@7
E8
F9
F
0
1
'2
(3
94
:5
?6
@7
E8
F9
­

Players
Qlayer_metrics
Rnon_trainable_variables
regularization_losses
	variables
trainable_variables
Smetrics
Tlayer_regularization_losses
 
\
U_rng
Vregularization_losses
W	variables
Xtrainable_variables
Y	keras_api
\
Z_rng
[regularization_losses
\	variables
]trainable_variables
^	keras_api
R
_regularization_losses
`	variables
atrainable_variables
b	keras_api
 
 
 
­

clayers
dlayer_metrics
enon_trainable_variables
regularization_losses
	variables
trainable_variables
fmetrics
glayer_regularization_losses
\Z
VARIABLE_VALUEconv2d_99/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_99/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
­

hlayers
ilayer_metrics
jnon_trainable_variables
regularization_losses
	variables
trainable_variables
kmetrics
llayer_regularization_losses
 
 
 
­

mlayers
nlayer_metrics
onon_trainable_variables
regularization_losses
 	variables
!trainable_variables
pmetrics
qlayer_regularization_losses
 
 
 
­

rlayers
slayer_metrics
tnon_trainable_variables
#regularization_losses
$	variables
%trainable_variables
umetrics
vlayer_regularization_losses
][
VARIABLE_VALUEconv2d_100/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_100/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

'0
(1

'0
(1
­

wlayers
xlayer_metrics
ynon_trainable_variables
)regularization_losses
*	variables
+trainable_variables
zmetrics
{layer_regularization_losses
 
 
 
?

|layers
}layer_metrics
~non_trainable_variables
-regularization_losses
.	variables
/trainable_variables
metrics
 layer_regularization_losses
 
 
 
²
layers
layer_metrics
non_trainable_variables
1regularization_losses
2	variables
3trainable_variables
metrics
 layer_regularization_losses
 
 
 
²
layers
layer_metrics
non_trainable_variables
5regularization_losses
6	variables
7trainable_variables
metrics
 layer_regularization_losses
\Z
VARIABLE_VALUEdense_198/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_198/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

90
:1

90
:1
²
layers
layer_metrics
non_trainable_variables
;regularization_losses
<	variables
=trainable_variables
metrics
 layer_regularization_losses
\Z
VARIABLE_VALUEdense_199/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_199/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
@1

?0
@1
²
layers
layer_metrics
non_trainable_variables
Aregularization_losses
B	variables
Ctrainable_variables
metrics
 layer_regularization_losses
\Z
VARIABLE_VALUEdense_200/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_200/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

E0
F1

E0
F1
²
layers
layer_metrics
non_trainable_variables
Gregularization_losses
H	variables
Itrainable_variables
metrics
 layer_regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
N
0
1
2
3
4
5
6
7
	8

9
10
 
 

0
1
 


_state_var
 
 
 
²
layers
layer_metrics
non_trainable_variables
Vregularization_losses
W	variables
Xtrainable_variables
 metrics
 ‘layer_regularization_losses

’
_state_var
 
 
 
²
£layers
€layer_metrics
₯non_trainable_variables
[regularization_losses
\	variables
]trainable_variables
¦metrics
 §layer_regularization_losses
 
 
 
²
¨layers
©layer_metrics
ͺnon_trainable_variables
_regularization_losses
`	variables
atrainable_variables
«metrics
 ¬layer_regularization_losses

0
1
2
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

­total

?count
―	variables
°	keras_api
I

±total

²count
³
_fn_kwargs
΄	variables
΅	keras_api
XV
VARIABLE_VALUEVariable:layer-0/layer-0/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
 
 
ZX
VARIABLE_VALUE
Variable_1:layer-0/layer-1/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

­0
?1

―	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

±0
²1

΄	variables
}
VARIABLE_VALUEAdam/conv2d_99/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_99/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/conv2d_100/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_100/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_198/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_198/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_199/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_199/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_200/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_200/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_99/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_99/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/conv2d_100/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_100/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_198/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_198/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_199/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_199/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_200/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_200/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
£
0serving_default_Training_Data_Augmentation_inputPlaceholder*/
_output_shapes
:?????????*
dtype0*$
shape:?????????

StatefulPartitionedCallStatefulPartitionedCall0serving_default_Training_Data_Augmentation_inputconv2d_99/kernelconv2d_99/biasconv2d_100/kernelconv2d_100/biasdense_198/kerneldense_198/biasdense_199/kerneldense_199/biasdense_200/kerneldense_200/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *.
f)R'
%__inference_signature_wrapper_1715940
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_99/kernel/Read/ReadVariableOp"conv2d_99/bias/Read/ReadVariableOp%conv2d_100/kernel/Read/ReadVariableOp#conv2d_100/bias/Read/ReadVariableOp$dense_198/kernel/Read/ReadVariableOp"dense_198/bias/Read/ReadVariableOp$dense_199/kernel/Read/ReadVariableOp"dense_199/bias/Read/ReadVariableOp$dense_200/kernel/Read/ReadVariableOp"dense_200/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOpVariable/Read/ReadVariableOpVariable_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp+Adam/conv2d_99/kernel/m/Read/ReadVariableOp)Adam/conv2d_99/bias/m/Read/ReadVariableOp,Adam/conv2d_100/kernel/m/Read/ReadVariableOp*Adam/conv2d_100/bias/m/Read/ReadVariableOp+Adam/dense_198/kernel/m/Read/ReadVariableOp)Adam/dense_198/bias/m/Read/ReadVariableOp+Adam/dense_199/kernel/m/Read/ReadVariableOp)Adam/dense_199/bias/m/Read/ReadVariableOp+Adam/dense_200/kernel/m/Read/ReadVariableOp)Adam/dense_200/bias/m/Read/ReadVariableOp+Adam/conv2d_99/kernel/v/Read/ReadVariableOp)Adam/conv2d_99/bias/v/Read/ReadVariableOp,Adam/conv2d_100/kernel/v/Read/ReadVariableOp*Adam/conv2d_100/bias/v/Read/ReadVariableOp+Adam/dense_198/kernel/v/Read/ReadVariableOp)Adam/dense_198/bias/v/Read/ReadVariableOp+Adam/dense_199/kernel/v/Read/ReadVariableOp)Adam/dense_199/bias/v/Read/ReadVariableOp+Adam/dense_200/kernel/v/Read/ReadVariableOp)Adam/dense_200/bias/v/Read/ReadVariableOpConst*6
Tin/
-2+			*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__traced_save_1716964
Λ
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_99/kernelconv2d_99/biasconv2d_100/kernelconv2d_100/biasdense_198/kerneldense_198/biasdense_199/kerneldense_199/biasdense_200/kerneldense_200/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateVariable
Variable_1totalcounttotal_1count_1Adam/conv2d_99/kernel/mAdam/conv2d_99/bias/mAdam/conv2d_100/kernel/mAdam/conv2d_100/bias/mAdam/dense_198/kernel/mAdam/dense_198/bias/mAdam/dense_199/kernel/mAdam/dense_199/bias/mAdam/dense_200/kernel/mAdam/dense_200/bias/mAdam/conv2d_99/kernel/vAdam/conv2d_99/bias/vAdam/conv2d_100/kernel/vAdam/conv2d_100/bias/vAdam/dense_198/kernel/vAdam/dense_198/bias/vAdam/dense_199/kernel/vAdam/dense_199/bias/vAdam/dense_200/kernel/vAdam/dense_200/bias/v*5
Tin.
,2**
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference__traced_restore_1717097ΝΊ
Ζ
g
H__inference_dropout_100_layer_call_and_return_conditional_losses_1715631

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nΫΆ?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeΌ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2
dropout/GreaterEqual/yΖ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????@2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????@2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
	
?
F__inference_conv2d_99_layer_call_and_return_conditional_losses_1715544

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp€
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????:::W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
Γ
d
H__inference_random_flip_layer_call_and_return_conditional_losses_1715246

inputs
identity}
IdentityIdentityinputs*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
»5
Ή
U__inference_DoubleConv_LessDense_avg_layer_call_and_return_conditional_losses_1715820

inputs&
"training_data_augmentation_1715786
conv2d_99_1715789
conv2d_99_1715791
conv2d_100_1715796
conv2d_100_1715798
dense_198_1715804
dense_198_1715806
dense_199_1715809
dense_199_1715811
dense_200_1715814
dense_200_1715816
identity’2Training_Data_Augmentation/StatefulPartitionedCall’"conv2d_100/StatefulPartitionedCall’!conv2d_99/StatefulPartitionedCall’!dense_198/StatefulPartitionedCall’!dense_199/StatefulPartitionedCall’!dense_200/StatefulPartitionedCall’#dropout_100/StatefulPartitionedCall’"dropout_99/StatefulPartitionedCallΠ
2Training_Data_Augmentation/StatefulPartitionedCallStatefulPartitionedCallinputs"training_data_augmentation_1715786*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *`
f[RY
W__inference_Training_Data_Augmentation_layer_call_and_return_conditional_losses_171547324
2Training_Data_Augmentation/StatefulPartitionedCallΩ
!conv2d_99/StatefulPartitionedCallStatefulPartitionedCall;Training_Data_Augmentation/StatefulPartitionedCall:output:0conv2d_99_1715789conv2d_99_1715791*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_99_layer_call_and_return_conditional_losses_17155442#
!conv2d_99/StatefulPartitionedCall₯
$average_pooling2d_44/PartitionedCallPartitionedCall*conv2d_99/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_average_pooling2d_44_layer_call_and_return_conditional_losses_17154962&
$average_pooling2d_44/PartitionedCall’
"dropout_99/StatefulPartitionedCallStatefulPartitionedCall-average_pooling2d_44/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_99_layer_call_and_return_conditional_losses_17155732$
"dropout_99/StatefulPartitionedCallΞ
"conv2d_100/StatefulPartitionedCallStatefulPartitionedCall+dropout_99/StatefulPartitionedCall:output:0conv2d_100_1715796conv2d_100_1715798*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_conv2d_100_layer_call_and_return_conditional_losses_17156022$
"conv2d_100/StatefulPartitionedCall¦
$average_pooling2d_45/PartitionedCallPartitionedCall+conv2d_100/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_average_pooling2d_45_layer_call_and_return_conditional_losses_17155082&
$average_pooling2d_45/PartitionedCallΚ
#dropout_100/StatefulPartitionedCallStatefulPartitionedCall-average_pooling2d_45/PartitionedCall:output:0#^dropout_99/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_100_layer_call_and_return_conditional_losses_17156312%
#dropout_100/StatefulPartitionedCall
flatten_66/PartitionedCallPartitionedCall,dropout_100/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_66_layer_call_and_return_conditional_losses_17156552
flatten_66/PartitionedCallΊ
!dense_198/StatefulPartitionedCallStatefulPartitionedCall#flatten_66/PartitionedCall:output:0dense_198_1715804dense_198_1715806*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_198_layer_call_and_return_conditional_losses_17156742#
!dense_198/StatefulPartitionedCallΐ
!dense_199/StatefulPartitionedCallStatefulPartitionedCall*dense_198/StatefulPartitionedCall:output:0dense_199_1715809dense_199_1715811*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_199_layer_call_and_return_conditional_losses_17157012#
!dense_199/StatefulPartitionedCallΐ
!dense_200/StatefulPartitionedCallStatefulPartitionedCall*dense_199/StatefulPartitionedCall:output:0dense_200_1715814dense_200_1715816*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_200_layer_call_and_return_conditional_losses_17157282#
!dense_200/StatefulPartitionedCall³
IdentityIdentity*dense_200/StatefulPartitionedCall:output:03^Training_Data_Augmentation/StatefulPartitionedCall#^conv2d_100/StatefulPartitionedCall"^conv2d_99/StatefulPartitionedCall"^dense_198/StatefulPartitionedCall"^dense_199/StatefulPartitionedCall"^dense_200/StatefulPartitionedCall$^dropout_100/StatefulPartitionedCall#^dropout_99/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:?????????:::::::::::2h
2Training_Data_Augmentation/StatefulPartitionedCall2Training_Data_Augmentation/StatefulPartitionedCall2H
"conv2d_100/StatefulPartitionedCall"conv2d_100/StatefulPartitionedCall2F
!conv2d_99/StatefulPartitionedCall!conv2d_99/StatefulPartitionedCall2F
!dense_198/StatefulPartitionedCall!dense_198/StatefulPartitionedCall2F
!dense_199/StatefulPartitionedCall!dense_199/StatefulPartitionedCall2F
!dense_200/StatefulPartitionedCall!dense_200/StatefulPartitionedCall2J
#dropout_100/StatefulPartitionedCall#dropout_100/StatefulPartitionedCall2H
"dropout_99/StatefulPartitionedCall"dropout_99/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
³
?
F__inference_dense_200_layer_call_and_return_conditional_losses_1716542

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????
2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:::O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
©F
¨
"__inference__wrapped_model_1715125$
 training_data_augmentation_inputE
Adoubleconv_lessdense_avg_conv2d_99_conv2d_readvariableop_resourceF
Bdoubleconv_lessdense_avg_conv2d_99_biasadd_readvariableop_resourceF
Bdoubleconv_lessdense_avg_conv2d_100_conv2d_readvariableop_resourceG
Cdoubleconv_lessdense_avg_conv2d_100_biasadd_readvariableop_resourceE
Adoubleconv_lessdense_avg_dense_198_matmul_readvariableop_resourceF
Bdoubleconv_lessdense_avg_dense_198_biasadd_readvariableop_resourceE
Adoubleconv_lessdense_avg_dense_199_matmul_readvariableop_resourceF
Bdoubleconv_lessdense_avg_dense_199_biasadd_readvariableop_resourceE
Adoubleconv_lessdense_avg_dense_200_matmul_readvariableop_resourceF
Bdoubleconv_lessdense_avg_dense_200_biasadd_readvariableop_resource
identityώ
8DoubleConv_LessDense_avg/conv2d_99/Conv2D/ReadVariableOpReadVariableOpAdoubleconv_lessdense_avg_conv2d_99_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02:
8DoubleConv_LessDense_avg/conv2d_99/Conv2D/ReadVariableOp§
)DoubleConv_LessDense_avg/conv2d_99/Conv2DConv2D training_data_augmentation_input@DoubleConv_LessDense_avg/conv2d_99/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2+
)DoubleConv_LessDense_avg/conv2d_99/Conv2Dυ
9DoubleConv_LessDense_avg/conv2d_99/BiasAdd/ReadVariableOpReadVariableOpBdoubleconv_lessdense_avg_conv2d_99_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02;
9DoubleConv_LessDense_avg/conv2d_99/BiasAdd/ReadVariableOp
*DoubleConv_LessDense_avg/conv2d_99/BiasAddBiasAdd2DoubleConv_LessDense_avg/conv2d_99/Conv2D:output:0ADoubleConv_LessDense_avg/conv2d_99/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2,
*DoubleConv_LessDense_avg/conv2d_99/BiasAddΙ
'DoubleConv_LessDense_avg/conv2d_99/ReluRelu3DoubleConv_LessDense_avg/conv2d_99/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2)
'DoubleConv_LessDense_avg/conv2d_99/Relu¦
5DoubleConv_LessDense_avg/average_pooling2d_44/AvgPoolAvgPool5DoubleConv_LessDense_avg/conv2d_99/Relu:activations:0*
T0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
27
5DoubleConv_LessDense_avg/average_pooling2d_44/AvgPoolβ
,DoubleConv_LessDense_avg/dropout_99/IdentityIdentity>DoubleConv_LessDense_avg/average_pooling2d_44/AvgPool:output:0*
T0*/
_output_shapes
:?????????@2.
,DoubleConv_LessDense_avg/dropout_99/Identity
9DoubleConv_LessDense_avg/conv2d_100/Conv2D/ReadVariableOpReadVariableOpBdoubleconv_lessdense_avg_conv2d_100_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02;
9DoubleConv_LessDense_avg/conv2d_100/Conv2D/ReadVariableOpΏ
*DoubleConv_LessDense_avg/conv2d_100/Conv2DConv2D5DoubleConv_LessDense_avg/dropout_99/Identity:output:0ADoubleConv_LessDense_avg/conv2d_100/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2,
*DoubleConv_LessDense_avg/conv2d_100/Conv2Dψ
:DoubleConv_LessDense_avg/conv2d_100/BiasAdd/ReadVariableOpReadVariableOpCdoubleconv_lessdense_avg_conv2d_100_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02<
:DoubleConv_LessDense_avg/conv2d_100/BiasAdd/ReadVariableOp
+DoubleConv_LessDense_avg/conv2d_100/BiasAddBiasAdd3DoubleConv_LessDense_avg/conv2d_100/Conv2D:output:0BDoubleConv_LessDense_avg/conv2d_100/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2-
+DoubleConv_LessDense_avg/conv2d_100/BiasAddΜ
(DoubleConv_LessDense_avg/conv2d_100/ReluRelu4DoubleConv_LessDense_avg/conv2d_100/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2*
(DoubleConv_LessDense_avg/conv2d_100/Relu§
5DoubleConv_LessDense_avg/average_pooling2d_45/AvgPoolAvgPool6DoubleConv_LessDense_avg/conv2d_100/Relu:activations:0*
T0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
27
5DoubleConv_LessDense_avg/average_pooling2d_45/AvgPoolδ
-DoubleConv_LessDense_avg/dropout_100/IdentityIdentity>DoubleConv_LessDense_avg/average_pooling2d_45/AvgPool:output:0*
T0*/
_output_shapes
:?????????@2/
-DoubleConv_LessDense_avg/dropout_100/Identity§
)DoubleConv_LessDense_avg/flatten_66/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 	  2+
)DoubleConv_LessDense_avg/flatten_66/Const
+DoubleConv_LessDense_avg/flatten_66/ReshapeReshape6DoubleConv_LessDense_avg/dropout_100/Identity:output:02DoubleConv_LessDense_avg/flatten_66/Const:output:0*
T0*(
_output_shapes
:?????????2-
+DoubleConv_LessDense_avg/flatten_66/Reshapeψ
8DoubleConv_LessDense_avg/dense_198/MatMul/ReadVariableOpReadVariableOpAdoubleconv_lessdense_avg_dense_198_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02:
8DoubleConv_LessDense_avg/dense_198/MatMul/ReadVariableOp
)DoubleConv_LessDense_avg/dense_198/MatMulMatMul4DoubleConv_LessDense_avg/flatten_66/Reshape:output:0@DoubleConv_LessDense_avg/dense_198/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2+
)DoubleConv_LessDense_avg/dense_198/MatMulφ
9DoubleConv_LessDense_avg/dense_198/BiasAdd/ReadVariableOpReadVariableOpBdoubleconv_lessdense_avg_dense_198_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02;
9DoubleConv_LessDense_avg/dense_198/BiasAdd/ReadVariableOp
*DoubleConv_LessDense_avg/dense_198/BiasAddBiasAdd3DoubleConv_LessDense_avg/dense_198/MatMul:product:0ADoubleConv_LessDense_avg/dense_198/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2,
*DoubleConv_LessDense_avg/dense_198/BiasAddΒ
'DoubleConv_LessDense_avg/dense_198/ReluRelu3DoubleConv_LessDense_avg/dense_198/BiasAdd:output:0*
T0*(
_output_shapes
:?????????2)
'DoubleConv_LessDense_avg/dense_198/Reluχ
8DoubleConv_LessDense_avg/dense_199/MatMul/ReadVariableOpReadVariableOpAdoubleconv_lessdense_avg_dense_199_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02:
8DoubleConv_LessDense_avg/dense_199/MatMul/ReadVariableOp
)DoubleConv_LessDense_avg/dense_199/MatMulMatMul5DoubleConv_LessDense_avg/dense_198/Relu:activations:0@DoubleConv_LessDense_avg/dense_199/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2+
)DoubleConv_LessDense_avg/dense_199/MatMulυ
9DoubleConv_LessDense_avg/dense_199/BiasAdd/ReadVariableOpReadVariableOpBdoubleconv_lessdense_avg_dense_199_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02;
9DoubleConv_LessDense_avg/dense_199/BiasAdd/ReadVariableOp
*DoubleConv_LessDense_avg/dense_199/BiasAddBiasAdd3DoubleConv_LessDense_avg/dense_199/MatMul:product:0ADoubleConv_LessDense_avg/dense_199/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2,
*DoubleConv_LessDense_avg/dense_199/BiasAddΑ
'DoubleConv_LessDense_avg/dense_199/ReluRelu3DoubleConv_LessDense_avg/dense_199/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2)
'DoubleConv_LessDense_avg/dense_199/Reluφ
8DoubleConv_LessDense_avg/dense_200/MatMul/ReadVariableOpReadVariableOpAdoubleconv_lessdense_avg_dense_200_matmul_readvariableop_resource*
_output_shapes

:@
*
dtype02:
8DoubleConv_LessDense_avg/dense_200/MatMul/ReadVariableOp
)DoubleConv_LessDense_avg/dense_200/MatMulMatMul5DoubleConv_LessDense_avg/dense_199/Relu:activations:0@DoubleConv_LessDense_avg/dense_200/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2+
)DoubleConv_LessDense_avg/dense_200/MatMulυ
9DoubleConv_LessDense_avg/dense_200/BiasAdd/ReadVariableOpReadVariableOpBdoubleconv_lessdense_avg_dense_200_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02;
9DoubleConv_LessDense_avg/dense_200/BiasAdd/ReadVariableOp
*DoubleConv_LessDense_avg/dense_200/BiasAddBiasAdd3DoubleConv_LessDense_avg/dense_200/MatMul:product:0ADoubleConv_LessDense_avg/dense_200/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2,
*DoubleConv_LessDense_avg/dense_200/BiasAddΚ
*DoubleConv_LessDense_avg/dense_200/SoftmaxSoftmax3DoubleConv_LessDense_avg/dense_200/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2,
*DoubleConv_LessDense_avg/dense_200/Softmax
IdentityIdentity4DoubleConv_LessDense_avg/dense_200/Softmax:softmax:0*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????:::::::::::q m
/
_output_shapes
:?????????
:
_user_specified_name" Training_Data_Augmentation_input
Β
M
1__inference_random_rotation_layer_call_fn_1716781

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_random_rotation_layer_call_and_return_conditional_losses_17154062
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
ι
w
1__inference_random_rotation_layer_call_fn_1716776

inputs
unknown
identity’StatefulPartitionedCallτ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_random_rotation_layer_call_and_return_conditional_losses_17154022
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????:22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
³
?
F__inference_dense_200_layer_call_and_return_conditional_losses_1715728

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????
2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:::O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
²
½
L__inference_random_rotation_layer_call_and_return_conditional_losses_1716765

inputs-
)stateful_uniform_statefuluniform_resource
identity’ stateful_uniform/StatefulUniformD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2β
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2μ
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1^
CastCaststrided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
Castx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2μ
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2b
Cast_1Caststrided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
Cast_1~
stateful_uniform/shapePackstrided_slice:output:0*
N*
T0*
_output_shapes
:2
stateful_uniform/shapeq
stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *|Ω Ώ2
stateful_uniform/minq
stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *|Ω ?2
stateful_uniform/max
*stateful_uniform/StatefulUniform/algorithmConst*
_output_shapes
: *
dtype0	*
value	B	 R2,
*stateful_uniform/StatefulUniform/algorithm
 stateful_uniform/StatefulUniformStatefulUniform)stateful_uniform_statefuluniform_resource3stateful_uniform/StatefulUniform/algorithm:output:0stateful_uniform/shape:output:0*#
_output_shapes
:?????????*
shape_dtype02"
 stateful_uniform/StatefulUniform
stateful_uniform/subSubstateful_uniform/max:output:0stateful_uniform/min:output:0*
T0*
_output_shapes
: 2
stateful_uniform/sub¦
stateful_uniform/mulMul)stateful_uniform/StatefulUniform:output:0stateful_uniform/sub:z:0*
T0*#
_output_shapes
:?????????2
stateful_uniform/mul
stateful_uniformAddstateful_uniform/mul:z:0stateful_uniform/min:output:0*
T0*#
_output_shapes
:?????????2
stateful_uniforms
rotation_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
rotation_matrix/sub/y~
rotation_matrix/subSub
Cast_1:y:0rotation_matrix/sub/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/subu
rotation_matrix/CosCosstateful_uniform:z:0*
T0*#
_output_shapes
:?????????2
rotation_matrix/Cosw
rotation_matrix/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
rotation_matrix/sub_1/y
rotation_matrix/sub_1Sub
Cast_1:y:0 rotation_matrix/sub_1/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/sub_1
rotation_matrix/mulMulrotation_matrix/Cos:y:0rotation_matrix/sub_1:z:0*
T0*#
_output_shapes
:?????????2
rotation_matrix/mulu
rotation_matrix/SinSinstateful_uniform:z:0*
T0*#
_output_shapes
:?????????2
rotation_matrix/Sinw
rotation_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
rotation_matrix/sub_2/y
rotation_matrix/sub_2SubCast:y:0 rotation_matrix/sub_2/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/sub_2
rotation_matrix/mul_1Mulrotation_matrix/Sin:y:0rotation_matrix/sub_2:z:0*
T0*#
_output_shapes
:?????????2
rotation_matrix/mul_1
rotation_matrix/sub_3Subrotation_matrix/mul:z:0rotation_matrix/mul_1:z:0*
T0*#
_output_shapes
:?????????2
rotation_matrix/sub_3
rotation_matrix/sub_4Subrotation_matrix/sub:z:0rotation_matrix/sub_3:z:0*
T0*#
_output_shapes
:?????????2
rotation_matrix/sub_4{
rotation_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
rotation_matrix/truediv/yͺ
rotation_matrix/truedivRealDivrotation_matrix/sub_4:z:0"rotation_matrix/truediv/y:output:0*
T0*#
_output_shapes
:?????????2
rotation_matrix/truedivw
rotation_matrix/sub_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
rotation_matrix/sub_5/y
rotation_matrix/sub_5SubCast:y:0 rotation_matrix/sub_5/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/sub_5y
rotation_matrix/Sin_1Sinstateful_uniform:z:0*
T0*#
_output_shapes
:?????????2
rotation_matrix/Sin_1w
rotation_matrix/sub_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
rotation_matrix/sub_6/y
rotation_matrix/sub_6Sub
Cast_1:y:0 rotation_matrix/sub_6/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/sub_6
rotation_matrix/mul_2Mulrotation_matrix/Sin_1:y:0rotation_matrix/sub_6:z:0*
T0*#
_output_shapes
:?????????2
rotation_matrix/mul_2y
rotation_matrix/Cos_1Cosstateful_uniform:z:0*
T0*#
_output_shapes
:?????????2
rotation_matrix/Cos_1w
rotation_matrix/sub_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
rotation_matrix/sub_7/y
rotation_matrix/sub_7SubCast:y:0 rotation_matrix/sub_7/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/sub_7
rotation_matrix/mul_3Mulrotation_matrix/Cos_1:y:0rotation_matrix/sub_7:z:0*
T0*#
_output_shapes
:?????????2
rotation_matrix/mul_3
rotation_matrix/addAddV2rotation_matrix/mul_2:z:0rotation_matrix/mul_3:z:0*
T0*#
_output_shapes
:?????????2
rotation_matrix/add
rotation_matrix/sub_8Subrotation_matrix/sub_5:z:0rotation_matrix/add:z:0*
T0*#
_output_shapes
:?????????2
rotation_matrix/sub_8
rotation_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
rotation_matrix/truediv_1/y°
rotation_matrix/truediv_1RealDivrotation_matrix/sub_8:z:0$rotation_matrix/truediv_1/y:output:0*
T0*#
_output_shapes
:?????????2
rotation_matrix/truediv_1r
rotation_matrix/ShapeShapestateful_uniform:z:0*
T0*
_output_shapes
:2
rotation_matrix/Shape
#rotation_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#rotation_matrix/strided_slice/stack
%rotation_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%rotation_matrix/strided_slice/stack_1
%rotation_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%rotation_matrix/strided_slice/stack_2Β
rotation_matrix/strided_sliceStridedSlicerotation_matrix/Shape:output:0,rotation_matrix/strided_slice/stack:output:0.rotation_matrix/strided_slice/stack_1:output:0.rotation_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
rotation_matrix/strided_slicey
rotation_matrix/Cos_2Cosstateful_uniform:z:0*
T0*#
_output_shapes
:?????????2
rotation_matrix/Cos_2
%rotation_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_1/stack£
'rotation_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_1/stack_1£
'rotation_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_1/stack_2χ
rotation_matrix/strided_slice_1StridedSlicerotation_matrix/Cos_2:y:0.rotation_matrix/strided_slice_1/stack:output:00rotation_matrix/strided_slice_1/stack_1:output:00rotation_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_1y
rotation_matrix/Sin_2Sinstateful_uniform:z:0*
T0*#
_output_shapes
:?????????2
rotation_matrix/Sin_2
%rotation_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_2/stack£
'rotation_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_2/stack_1£
'rotation_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_2/stack_2χ
rotation_matrix/strided_slice_2StridedSlicerotation_matrix/Sin_2:y:0.rotation_matrix/strided_slice_2/stack:output:00rotation_matrix/strided_slice_2/stack_1:output:00rotation_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_2
rotation_matrix/NegNeg(rotation_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????2
rotation_matrix/Neg
%rotation_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_3/stack£
'rotation_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_3/stack_1£
'rotation_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_3/stack_2ω
rotation_matrix/strided_slice_3StridedSlicerotation_matrix/truediv:z:0.rotation_matrix/strided_slice_3/stack:output:00rotation_matrix/strided_slice_3/stack_1:output:00rotation_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_3y
rotation_matrix/Sin_3Sinstateful_uniform:z:0*
T0*#
_output_shapes
:?????????2
rotation_matrix/Sin_3
%rotation_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_4/stack£
'rotation_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_4/stack_1£
'rotation_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_4/stack_2χ
rotation_matrix/strided_slice_4StridedSlicerotation_matrix/Sin_3:y:0.rotation_matrix/strided_slice_4/stack:output:00rotation_matrix/strided_slice_4/stack_1:output:00rotation_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_4y
rotation_matrix/Cos_3Cosstateful_uniform:z:0*
T0*#
_output_shapes
:?????????2
rotation_matrix/Cos_3
%rotation_matrix/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_5/stack£
'rotation_matrix/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_5/stack_1£
'rotation_matrix/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_5/stack_2χ
rotation_matrix/strided_slice_5StridedSlicerotation_matrix/Cos_3:y:0.rotation_matrix/strided_slice_5/stack:output:00rotation_matrix/strided_slice_5/stack_1:output:00rotation_matrix/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_5
%rotation_matrix/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_6/stack£
'rotation_matrix/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_6/stack_1£
'rotation_matrix/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_6/stack_2ϋ
rotation_matrix/strided_slice_6StridedSlicerotation_matrix/truediv_1:z:0.rotation_matrix/strided_slice_6/stack:output:00rotation_matrix/strided_slice_6/stack_1:output:00rotation_matrix/strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_6|
rotation_matrix/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
rotation_matrix/zeros/mul/y¬
rotation_matrix/zeros/mulMul&rotation_matrix/strided_slice:output:0$rotation_matrix/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/zeros/mul
rotation_matrix/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :θ2
rotation_matrix/zeros/Less/y§
rotation_matrix/zeros/LessLessrotation_matrix/zeros/mul:z:0%rotation_matrix/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/zeros/Less
rotation_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2 
rotation_matrix/zeros/packed/1Γ
rotation_matrix/zeros/packedPack&rotation_matrix/strided_slice:output:0'rotation_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
rotation_matrix/zeros/packed
rotation_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
rotation_matrix/zeros/Const΅
rotation_matrix/zerosFill%rotation_matrix/zeros/packed:output:0$rotation_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:?????????2
rotation_matrix/zeros|
rotation_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
rotation_matrix/concat/axis¨
rotation_matrix/concatConcatV2(rotation_matrix/strided_slice_1:output:0rotation_matrix/Neg:y:0(rotation_matrix/strided_slice_3:output:0(rotation_matrix/strided_slice_4:output:0(rotation_matrix/strided_slice_5:output:0(rotation_matrix/strided_slice_6:output:0rotation_matrix/zeros:output:0$rotation_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
rotation_matrix/concatX
transform/ShapeShapeinputs*
T0*
_output_shapes
:2
transform/Shape
transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
transform/strided_slice/stack
transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
transform/strided_slice/stack_1
transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
transform/strided_slice/stack_2
transform/strided_sliceStridedSlicetransform/Shape:output:0&transform/strided_slice/stack:output:0(transform/strided_slice/stack_1:output:0(transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
transform/strided_slice¨
$transform/ImageProjectiveTransformV2ImageProjectiveTransformV2inputsrotation_matrix/concat:output:0 transform/strided_slice:output:0*/
_output_shapes
:?????????*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR2&
$transform/ImageProjectiveTransformV2Έ
IdentityIdentity9transform/ImageProjectiveTransformV2:transformed_images:0!^stateful_uniform/StatefulUniform*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????:2D
 stateful_uniform/StatefulUniform stateful_uniform/StatefulUniform:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
²
f
-__inference_random_flip_layer_call_fn_1716654

inputs
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_random_flip_layer_call_and_return_conditional_losses_17152372
StatefulPartitionedCall±
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????22
StatefulPartitionedCallStatefulPartitionedCall:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
΄
?
F__inference_dense_198_layer_call_and_return_conditional_losses_1715674

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:?????????2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:::P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
Ψ
~
W__inference_Training_Data_Augmentation_layer_call_and_return_conditional_losses_1715461
random_flip_input
identityρ
random_flip/PartitionedCallPartitionedCallrandom_flip_input*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_random_flip_layer_call_and_return_conditional_losses_17152842
random_flip/PartitionedCall
random_rotation/PartitionedCallPartitionedCall$random_flip/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_random_rotation_layer_call_and_return_conditional_losses_17154062!
random_rotation/PartitionedCall
random_contrast/PartitionedCallPartitionedCall(random_rotation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_random_contrast_layer_call_and_return_conditional_losses_17154402!
random_contrast/PartitionedCall
IdentityIdentity(random_contrast/PartitionedCall:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:b ^
/
_output_shapes
:?????????
+
_user_specified_namerandom_flip_input
ΑV
ο
 __inference__traced_save_1716964
file_prefix/
+savev2_conv2d_99_kernel_read_readvariableop-
)savev2_conv2d_99_bias_read_readvariableop0
,savev2_conv2d_100_kernel_read_readvariableop.
*savev2_conv2d_100_bias_read_readvariableop/
+savev2_dense_198_kernel_read_readvariableop-
)savev2_dense_198_bias_read_readvariableop/
+savev2_dense_199_kernel_read_readvariableop-
)savev2_dense_199_bias_read_readvariableop/
+savev2_dense_200_kernel_read_readvariableop-
)savev2_dense_200_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop'
#savev2_variable_read_readvariableop	)
%savev2_variable_1_read_readvariableop	$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop6
2savev2_adam_conv2d_99_kernel_m_read_readvariableop4
0savev2_adam_conv2d_99_bias_m_read_readvariableop7
3savev2_adam_conv2d_100_kernel_m_read_readvariableop5
1savev2_adam_conv2d_100_bias_m_read_readvariableop6
2savev2_adam_dense_198_kernel_m_read_readvariableop4
0savev2_adam_dense_198_bias_m_read_readvariableop6
2savev2_adam_dense_199_kernel_m_read_readvariableop4
0savev2_adam_dense_199_bias_m_read_readvariableop6
2savev2_adam_dense_200_kernel_m_read_readvariableop4
0savev2_adam_dense_200_bias_m_read_readvariableop6
2savev2_adam_conv2d_99_kernel_v_read_readvariableop4
0savev2_adam_conv2d_99_bias_v_read_readvariableop7
3savev2_adam_conv2d_100_kernel_v_read_readvariableop5
1savev2_adam_conv2d_100_bias_v_read_readvariableop6
2savev2_adam_dense_198_kernel_v_read_readvariableop4
0savev2_adam_dense_198_bias_v_read_readvariableop6
2savev2_adam_dense_199_kernel_v_read_readvariableop4
0savev2_adam_dense_199_bias_v_read_readvariableop6
2savev2_adam_dense_200_kernel_v_read_readvariableop4
0savev2_adam_dense_200_bias_v_read_readvariableop
savev2_const

identity_1’MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_d936c07952c0488bb5fd0c807bfbe4cc/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameψ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:**
dtype0*
valueBύ*B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB:layer-0/layer-0/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUEB:layer-0/layer-1/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesά
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:**
dtype0*g
value^B\*B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesΑ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_99_kernel_read_readvariableop)savev2_conv2d_99_bias_read_readvariableop,savev2_conv2d_100_kernel_read_readvariableop*savev2_conv2d_100_bias_read_readvariableop+savev2_dense_198_kernel_read_readvariableop)savev2_dense_198_bias_read_readvariableop+savev2_dense_199_kernel_read_readvariableop)savev2_dense_199_bias_read_readvariableop+savev2_dense_200_kernel_read_readvariableop)savev2_dense_200_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop#savev2_variable_read_readvariableop%savev2_variable_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop2savev2_adam_conv2d_99_kernel_m_read_readvariableop0savev2_adam_conv2d_99_bias_m_read_readvariableop3savev2_adam_conv2d_100_kernel_m_read_readvariableop1savev2_adam_conv2d_100_bias_m_read_readvariableop2savev2_adam_dense_198_kernel_m_read_readvariableop0savev2_adam_dense_198_bias_m_read_readvariableop2savev2_adam_dense_199_kernel_m_read_readvariableop0savev2_adam_dense_199_bias_m_read_readvariableop2savev2_adam_dense_200_kernel_m_read_readvariableop0savev2_adam_dense_200_bias_m_read_readvariableop2savev2_adam_conv2d_99_kernel_v_read_readvariableop0savev2_adam_conv2d_99_bias_v_read_readvariableop3savev2_adam_conv2d_100_kernel_v_read_readvariableop1savev2_adam_conv2d_100_bias_v_read_readvariableop2savev2_adam_dense_198_kernel_v_read_readvariableop0savev2_adam_dense_198_bias_v_read_readvariableop2savev2_adam_dense_199_kernel_v_read_readvariableop0savev2_adam_dense_199_bias_v_read_readvariableop2savev2_adam_dense_200_kernel_v_read_readvariableop0savev2_adam_dense_200_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *8
dtypes.
,2*			2
SaveV2Ί
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes‘
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*γ
_input_shapesΡ
Ξ: :@:@:@@:@:
::	@:@:@
:
: : : : : ::: : : : :@:@:@@:@:
::	@:@:@
:
:@:@:@@:@:
::	@:@:@
:
: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	@: 

_output_shapes
:@:$	 

_output_shapes

:@
: 


_output_shapes
:
:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	@: 

_output_shapes
:@:$ 

_output_shapes

:@
: 

_output_shapes
:
:, (
&
_output_shapes
:@: !

_output_shapes
:@:,"(
&
_output_shapes
:@@: #

_output_shapes
:@:&$"
 
_output_shapes
:
:!%

_output_shapes	
::%&!

_output_shapes
:	@: '

_output_shapes
:@:$( 

_output_shapes

:@
: )

_output_shapes
:
:*

_output_shapes
: 
ω
c
<__inference_Training_Data_Augmentation_layer_call_fn_1715490
random_flip_input
identityθ
PartitionedCallPartitionedCallrandom_flip_input*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *`
f[RY
W__inference_Training_Data_Augmentation_layer_call_and_return_conditional_losses_17154872
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:b ^
/
_output_shapes
:?????????
+
_user_specified_namerandom_flip_input
ν¬

#__inference__traced_restore_1717097
file_prefix%
!assignvariableop_conv2d_99_kernel%
!assignvariableop_1_conv2d_99_bias(
$assignvariableop_2_conv2d_100_kernel&
"assignvariableop_3_conv2d_100_bias'
#assignvariableop_4_dense_198_kernel%
!assignvariableop_5_dense_198_bias'
#assignvariableop_6_dense_199_kernel%
!assignvariableop_7_dense_199_bias'
#assignvariableop_8_dense_200_kernel%
!assignvariableop_9_dense_200_bias!
assignvariableop_10_adam_iter#
assignvariableop_11_adam_beta_1#
assignvariableop_12_adam_beta_2"
assignvariableop_13_adam_decay*
&assignvariableop_14_adam_learning_rate 
assignvariableop_15_variable"
assignvariableop_16_variable_1
assignvariableop_17_total
assignvariableop_18_count
assignvariableop_19_total_1
assignvariableop_20_count_1/
+assignvariableop_21_adam_conv2d_99_kernel_m-
)assignvariableop_22_adam_conv2d_99_bias_m0
,assignvariableop_23_adam_conv2d_100_kernel_m.
*assignvariableop_24_adam_conv2d_100_bias_m/
+assignvariableop_25_adam_dense_198_kernel_m-
)assignvariableop_26_adam_dense_198_bias_m/
+assignvariableop_27_adam_dense_199_kernel_m-
)assignvariableop_28_adam_dense_199_bias_m/
+assignvariableop_29_adam_dense_200_kernel_m-
)assignvariableop_30_adam_dense_200_bias_m/
+assignvariableop_31_adam_conv2d_99_kernel_v-
)assignvariableop_32_adam_conv2d_99_bias_v0
,assignvariableop_33_adam_conv2d_100_kernel_v.
*assignvariableop_34_adam_conv2d_100_bias_v/
+assignvariableop_35_adam_dense_198_kernel_v-
)assignvariableop_36_adam_dense_198_bias_v/
+assignvariableop_37_adam_dense_199_kernel_v-
)assignvariableop_38_adam_dense_199_bias_v/
+assignvariableop_39_adam_dense_200_kernel_v-
)assignvariableop_40_adam_dense_200_bias_v
identity_42’AssignVariableOp’AssignVariableOp_1’AssignVariableOp_10’AssignVariableOp_11’AssignVariableOp_12’AssignVariableOp_13’AssignVariableOp_14’AssignVariableOp_15’AssignVariableOp_16’AssignVariableOp_17’AssignVariableOp_18’AssignVariableOp_19’AssignVariableOp_2’AssignVariableOp_20’AssignVariableOp_21’AssignVariableOp_22’AssignVariableOp_23’AssignVariableOp_24’AssignVariableOp_25’AssignVariableOp_26’AssignVariableOp_27’AssignVariableOp_28’AssignVariableOp_29’AssignVariableOp_3’AssignVariableOp_30’AssignVariableOp_31’AssignVariableOp_32’AssignVariableOp_33’AssignVariableOp_34’AssignVariableOp_35’AssignVariableOp_36’AssignVariableOp_37’AssignVariableOp_38’AssignVariableOp_39’AssignVariableOp_4’AssignVariableOp_40’AssignVariableOp_5’AssignVariableOp_6’AssignVariableOp_7’AssignVariableOp_8’AssignVariableOp_9ώ
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:**
dtype0*
valueBύ*B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB:layer-0/layer-0/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUEB:layer-0/layer-1/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesβ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:**
dtype0*g
value^B\*B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Ύ
_output_shapes«
¨::::::::::::::::::::::::::::::::::::::::::*8
dtypes.
,2*			2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity 
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_99_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¦
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_99_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2©
AssignVariableOp_2AssignVariableOp$assignvariableop_2_conv2d_100_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3§
AssignVariableOp_3AssignVariableOp"assignvariableop_3_conv2d_100_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4¨
AssignVariableOp_4AssignVariableOp#assignvariableop_4_dense_198_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5¦
AssignVariableOp_5AssignVariableOp!assignvariableop_5_dense_198_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6¨
AssignVariableOp_6AssignVariableOp#assignvariableop_6_dense_199_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7¦
AssignVariableOp_7AssignVariableOp!assignvariableop_7_dense_199_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8¨
AssignVariableOp_8AssignVariableOp#assignvariableop_8_dense_200_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9¦
AssignVariableOp_9AssignVariableOp!assignvariableop_9_dense_200_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_10₯
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_iterIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11§
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_beta_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12§
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_beta_2Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13¦
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_decayIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp&assignvariableop_14_adam_learning_rateIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_15€
AssignVariableOp_15AssignVariableOpassignvariableop_15_variableIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_16¦
AssignVariableOp_16AssignVariableOpassignvariableop_16_variable_1Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17‘
AssignVariableOp_17AssignVariableOpassignvariableop_17_totalIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18‘
AssignVariableOp_18AssignVariableOpassignvariableop_18_countIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19£
AssignVariableOp_19AssignVariableOpassignvariableop_19_total_1Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20£
AssignVariableOp_20AssignVariableOpassignvariableop_20_count_1Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21³
AssignVariableOp_21AssignVariableOp+assignvariableop_21_adam_conv2d_99_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22±
AssignVariableOp_22AssignVariableOp)assignvariableop_22_adam_conv2d_99_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23΄
AssignVariableOp_23AssignVariableOp,assignvariableop_23_adam_conv2d_100_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24²
AssignVariableOp_24AssignVariableOp*assignvariableop_24_adam_conv2d_100_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25³
AssignVariableOp_25AssignVariableOp+assignvariableop_25_adam_dense_198_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26±
AssignVariableOp_26AssignVariableOp)assignvariableop_26_adam_dense_198_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27³
AssignVariableOp_27AssignVariableOp+assignvariableop_27_adam_dense_199_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28±
AssignVariableOp_28AssignVariableOp)assignvariableop_28_adam_dense_199_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29³
AssignVariableOp_29AssignVariableOp+assignvariableop_29_adam_dense_200_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30±
AssignVariableOp_30AssignVariableOp)assignvariableop_30_adam_dense_200_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31³
AssignVariableOp_31AssignVariableOp+assignvariableop_31_adam_conv2d_99_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32±
AssignVariableOp_32AssignVariableOp)assignvariableop_32_adam_conv2d_99_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33΄
AssignVariableOp_33AssignVariableOp,assignvariableop_33_adam_conv2d_100_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34²
AssignVariableOp_34AssignVariableOp*assignvariableop_34_adam_conv2d_100_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35³
AssignVariableOp_35AssignVariableOp+assignvariableop_35_adam_dense_198_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36±
AssignVariableOp_36AssignVariableOp)assignvariableop_36_adam_dense_198_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37³
AssignVariableOp_37AssignVariableOp+assignvariableop_37_adam_dense_199_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38±
AssignVariableOp_38AssignVariableOp)assignvariableop_38_adam_dense_199_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39³
AssignVariableOp_39AssignVariableOp+assignvariableop_39_adam_dense_200_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40±
AssignVariableOp_40AssignVariableOp)assignvariableop_40_adam_dense_200_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_409
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpδ
Identity_41Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_41Χ
Identity_42IdentityIdentity_41:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_42"#
identity_42Identity_42:output:0*»
_input_shapes©
¦: :::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ώ
c
G__inference_flatten_66_layer_call_and_return_conditional_losses_1715655

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 	  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:?????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
φ
h
L__inference_random_contrast_layer_call_and_return_conditional_losses_1716798

inputs
identityb
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
	

:__inference_DoubleConv_LessDense_avg_layer_call_fn_1716231

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity’StatefulPartitionedCallν
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *^
fYRW
U__inference_DoubleConv_LessDense_avg_layer_call_and_return_conditional_losses_17158822
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
Ί
I
-__inference_dropout_100_layer_call_fn_1716480

inputs
identityΞ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_100_layer_call_and_return_conditional_losses_17156362
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
κ
e
G__inference_dropout_99_layer_call_and_return_conditional_losses_1715578

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????@2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????@2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
Δ
e
,__inference_dropout_99_layer_call_fn_1716428

inputs
identity’StatefulPartitionedCallε
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_99_layer_call_and_return_conditional_losses_17155732
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs

s
W__inference_Training_Data_Augmentation_layer_call_and_return_conditional_losses_1716374

inputs
identityb
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs


W__inference_Training_Data_Augmentation_layer_call_and_return_conditional_losses_1715454
random_flip_input
random_rotation_1715420
identity’'random_contrast/StatefulPartitionedCall’#random_flip/StatefulPartitionedCall’'random_rotation/StatefulPartitionedCall
#random_flip/StatefulPartitionedCallStatefulPartitionedCallrandom_flip_input*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_random_flip_layer_call_and_return_conditional_losses_17152802%
#random_flip/StatefulPartitionedCallΚ
'random_rotation/StatefulPartitionedCallStatefulPartitionedCall,random_flip/StatefulPartitionedCall:output:0random_rotation_1715420*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_random_rotation_layer_call_and_return_conditional_losses_17154022)
'random_rotation/StatefulPartitionedCallΪ
'random_contrast/StatefulPartitionedCallStatefulPartitionedCall0random_rotation/StatefulPartitionedCall:output:0$^random_flip/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_random_contrast_layer_call_and_return_conditional_losses_17154362)
'random_contrast/StatefulPartitionedCall
IdentityIdentity0random_contrast/StatefulPartitionedCall:output:0(^random_contrast/StatefulPartitionedCall$^random_flip/StatefulPartitionedCall(^random_rotation/StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????:2R
'random_contrast/StatefulPartitionedCall'random_contrast/StatefulPartitionedCall2J
#random_flip/StatefulPartitionedCall#random_flip/StatefulPartitionedCall2R
'random_rotation/StatefulPartitionedCall'random_rotation/StatefulPartitionedCall:b ^
/
_output_shapes
:?????????
+
_user_specified_namerandom_flip_input
	
?
F__inference_conv2d_99_layer_call_and_return_conditional_losses_1716397

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp€
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????:::W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
Ζ
g
H__inference_dropout_100_layer_call_and_return_conditional_losses_1716465

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nΫΆ?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeΌ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2
dropout/GreaterEqual/yΖ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????@2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????@2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
α

+__inference_dense_200_layer_call_fn_1716551

inputs
unknown
	unknown_0
identity’StatefulPartitionedCallφ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_200_layer_call_and_return_conditional_losses_17157282
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
Β
M
1__inference_random_contrast_layer_call_fn_1716808

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_random_contrast_layer_call_and_return_conditional_losses_17154402
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
Ψ
X
<__inference_Training_Data_Augmentation_layer_call_fn_1716386

inputs
identityέ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *`
f[RY
W__inference_Training_Data_Augmentation_layer_call_and_return_conditional_losses_17154872
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
φ
h
L__inference_random_rotation_layer_call_and_return_conditional_losses_1716769

inputs
identityb
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
	
―
G__inference_conv2d_100_layer_call_and_return_conditional_losses_1715602

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp€
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@:::W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
Ώ
c
G__inference_flatten_66_layer_call_and_return_conditional_losses_1716486

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 	  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:?????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
φ
h
L__inference_random_rotation_layer_call_and_return_conditional_losses_1715406

inputs
identityb
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
»
k
L__inference_random_contrast_layer_call_and_return_conditional_losses_1715436

inputs
identityo
random_uniform/shapeConst*
_output_shapes
: *
dtype0*
valueB 2
random_uniform/shapem
random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2
random_uniform/minm
random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜ?2
random_uniform/max
random_uniform/RandomUniformRandomUniformrandom_uniform/shape:output:0*
T0*
_output_shapes
: *
dtype02
random_uniform/RandomUniform
random_uniform/subSubrandom_uniform/max:output:0random_uniform/min:output:0*
T0*
_output_shapes
: 2
random_uniform/sub
random_uniform/mulMul%random_uniform/RandomUniform:output:0random_uniform/sub:z:0*
T0*
_output_shapes
: 2
random_uniform/mul}
random_uniformAddrandom_uniform/mul:z:0random_uniform/min:output:0*
T0*
_output_shapes
: 2
random_uniform
adjust_contrastAdjustContrastv2inputsrandom_uniform:z:0*/
_output_shapes
:?????????2
adjust_contrast
adjust_contrast/IdentityIdentityadjust_contrast:output:0*
T0*/
_output_shapes
:?????????2
adjust_contrast/Identity}
IdentityIdentity!adjust_contrast/Identity:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
λ
f
H__inference_dropout_100_layer_call_and_return_conditional_losses_1716470

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????@2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????@2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
δ

W__inference_Training_Data_Augmentation_layer_call_and_return_conditional_losses_1715473

inputs
random_rotation_1715468
identity’'random_contrast/StatefulPartitionedCall’#random_flip/StatefulPartitionedCall’'random_rotation/StatefulPartitionedCallώ
#random_flip/StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_random_flip_layer_call_and_return_conditional_losses_17152802%
#random_flip/StatefulPartitionedCallΚ
'random_rotation/StatefulPartitionedCallStatefulPartitionedCall,random_flip/StatefulPartitionedCall:output:0random_rotation_1715468*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_random_rotation_layer_call_and_return_conditional_losses_17154022)
'random_rotation/StatefulPartitionedCallΪ
'random_contrast/StatefulPartitionedCallStatefulPartitionedCall0random_rotation/StatefulPartitionedCall:output:0$^random_flip/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_random_contrast_layer_call_and_return_conditional_losses_17154362)
'random_contrast/StatefulPartitionedCall
IdentityIdentity0random_contrast/StatefulPartitionedCall:output:0(^random_contrast/StatefulPartitionedCall$^random_flip/StatefulPartitionedCall(^random_rotation/StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????:2R
'random_contrast/StatefulPartitionedCall'random_contrast/StatefulPartitionedCall2J
#random_flip/StatefulPartitionedCall#random_flip/StatefulPartitionedCall2R
'random_rotation/StatefulPartitionedCall'random_rotation/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?	

:__inference_DoubleConv_LessDense_avg_layer_call_fn_1716206

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9
identity’StatefulPartitionedCallω
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *^
fYRW
U__inference_DoubleConv_LessDense_avg_layer_call_and_return_conditional_losses_17158202
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:?????????:::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
ς
d
H__inference_random_flip_layer_call_and_return_conditional_losses_1716583

inputs
identityb
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
ε

+__inference_dense_198_layer_call_fn_1716511

inputs
unknown
	unknown_0
identity’StatefulPartitionedCallχ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_198_layer_call_and_return_conditional_losses_17156742
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
§
I
-__inference_random_flip_layer_call_fn_1716659

inputs
identityι
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_random_flip_layer_call_and_return_conditional_losses_17152462
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
Ξ
j
1__inference_random_contrast_layer_call_fn_1716803

inputs
identity’StatefulPartitionedCallκ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_random_contrast_layer_call_and_return_conditional_losses_17154362
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
ͺ
H
,__inference_flatten_66_layer_call_fn_1716491

inputs
identityΖ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_66_layer_call_and_return_conditional_losses_17156552
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
	

%__inference_signature_wrapper_1715940$
 training_data_augmentation_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity’StatefulPartitionedCallΤ
StatefulPartitionedCallStatefulPartitionedCall training_data_augmentation_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__wrapped_model_17151252
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:q m
/
_output_shapes
:?????????
:
_user_specified_name" Training_Data_Augmentation_input
?
?
F__inference_dense_199_layer_call_and_return_conditional_losses_1716522

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:::P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
λ
f
H__inference_dropout_100_layer_call_and_return_conditional_losses_1715636

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????@2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????@2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs


,__inference_conv2d_100_layer_call_fn_1716453

inputs
unknown
	unknown_0
identity’StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_conv2d_100_layer_call_and_return_conditional_losses_17156022
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs

m
Q__inference_average_pooling2d_44_layer_call_and_return_conditional_losses_1715496

inputs
identityΆ
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
AvgPool
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs


<__inference_Training_Data_Augmentation_layer_call_fn_1716381

inputs
unknown
identity’StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *`
f[RY
W__inference_Training_Data_Augmentation_layer_call_and_return_conditional_losses_17154732
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????:22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
Ή
R
6__inference_average_pooling2d_44_layer_call_fn_1715502

inputs
identityς
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_average_pooling2d_44_layer_call_and_return_conditional_losses_17154962
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
κ
e
G__inference_dropout_99_layer_call_and_return_conditional_losses_1716423

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????@2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????@2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
»
k
L__inference_random_contrast_layer_call_and_return_conditional_losses_1716794

inputs
identityo
random_uniform/shapeConst*
_output_shapes
: *
dtype0*
valueB 2
random_uniform/shapem
random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2
random_uniform/minm
random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜ?2
random_uniform/max
random_uniform/RandomUniformRandomUniformrandom_uniform/shape:output:0*
T0*
_output_shapes
: *
dtype02
random_uniform/RandomUniform
random_uniform/subSubrandom_uniform/max:output:0random_uniform/min:output:0*
T0*
_output_shapes
: 2
random_uniform/sub
random_uniform/mulMul%random_uniform/RandomUniform:output:0random_uniform/sub:z:0*
T0*
_output_shapes
: 2
random_uniform/mul}
random_uniformAddrandom_uniform/mul:z:0random_uniform/min:output:0*
T0*
_output_shapes
: 2
random_uniform
adjust_contrastAdjustContrastv2inputsrandom_uniform:z:0*/
_output_shapes
:?????????2
adjust_contrast
adjust_contrast/IdentityIdentityadjust_contrast:output:0*
T0*/
_output_shapes
:?????????2
adjust_contrast/Identity}
IdentityIdentity!adjust_contrast/Identity:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
²
½
L__inference_random_rotation_layer_call_and_return_conditional_losses_1715402

inputs-
)stateful_uniform_statefuluniform_resource
identity’ stateful_uniform/StatefulUniformD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2β
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2μ
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1^
CastCaststrided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
Castx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2μ
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2b
Cast_1Caststrided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
Cast_1~
stateful_uniform/shapePackstrided_slice:output:0*
N*
T0*
_output_shapes
:2
stateful_uniform/shapeq
stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *|Ω Ώ2
stateful_uniform/minq
stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *|Ω ?2
stateful_uniform/max
*stateful_uniform/StatefulUniform/algorithmConst*
_output_shapes
: *
dtype0	*
value	B	 R2,
*stateful_uniform/StatefulUniform/algorithm
 stateful_uniform/StatefulUniformStatefulUniform)stateful_uniform_statefuluniform_resource3stateful_uniform/StatefulUniform/algorithm:output:0stateful_uniform/shape:output:0*#
_output_shapes
:?????????*
shape_dtype02"
 stateful_uniform/StatefulUniform
stateful_uniform/subSubstateful_uniform/max:output:0stateful_uniform/min:output:0*
T0*
_output_shapes
: 2
stateful_uniform/sub¦
stateful_uniform/mulMul)stateful_uniform/StatefulUniform:output:0stateful_uniform/sub:z:0*
T0*#
_output_shapes
:?????????2
stateful_uniform/mul
stateful_uniformAddstateful_uniform/mul:z:0stateful_uniform/min:output:0*
T0*#
_output_shapes
:?????????2
stateful_uniforms
rotation_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
rotation_matrix/sub/y~
rotation_matrix/subSub
Cast_1:y:0rotation_matrix/sub/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/subu
rotation_matrix/CosCosstateful_uniform:z:0*
T0*#
_output_shapes
:?????????2
rotation_matrix/Cosw
rotation_matrix/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
rotation_matrix/sub_1/y
rotation_matrix/sub_1Sub
Cast_1:y:0 rotation_matrix/sub_1/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/sub_1
rotation_matrix/mulMulrotation_matrix/Cos:y:0rotation_matrix/sub_1:z:0*
T0*#
_output_shapes
:?????????2
rotation_matrix/mulu
rotation_matrix/SinSinstateful_uniform:z:0*
T0*#
_output_shapes
:?????????2
rotation_matrix/Sinw
rotation_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
rotation_matrix/sub_2/y
rotation_matrix/sub_2SubCast:y:0 rotation_matrix/sub_2/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/sub_2
rotation_matrix/mul_1Mulrotation_matrix/Sin:y:0rotation_matrix/sub_2:z:0*
T0*#
_output_shapes
:?????????2
rotation_matrix/mul_1
rotation_matrix/sub_3Subrotation_matrix/mul:z:0rotation_matrix/mul_1:z:0*
T0*#
_output_shapes
:?????????2
rotation_matrix/sub_3
rotation_matrix/sub_4Subrotation_matrix/sub:z:0rotation_matrix/sub_3:z:0*
T0*#
_output_shapes
:?????????2
rotation_matrix/sub_4{
rotation_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
rotation_matrix/truediv/yͺ
rotation_matrix/truedivRealDivrotation_matrix/sub_4:z:0"rotation_matrix/truediv/y:output:0*
T0*#
_output_shapes
:?????????2
rotation_matrix/truedivw
rotation_matrix/sub_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
rotation_matrix/sub_5/y
rotation_matrix/sub_5SubCast:y:0 rotation_matrix/sub_5/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/sub_5y
rotation_matrix/Sin_1Sinstateful_uniform:z:0*
T0*#
_output_shapes
:?????????2
rotation_matrix/Sin_1w
rotation_matrix/sub_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
rotation_matrix/sub_6/y
rotation_matrix/sub_6Sub
Cast_1:y:0 rotation_matrix/sub_6/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/sub_6
rotation_matrix/mul_2Mulrotation_matrix/Sin_1:y:0rotation_matrix/sub_6:z:0*
T0*#
_output_shapes
:?????????2
rotation_matrix/mul_2y
rotation_matrix/Cos_1Cosstateful_uniform:z:0*
T0*#
_output_shapes
:?????????2
rotation_matrix/Cos_1w
rotation_matrix/sub_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
rotation_matrix/sub_7/y
rotation_matrix/sub_7SubCast:y:0 rotation_matrix/sub_7/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/sub_7
rotation_matrix/mul_3Mulrotation_matrix/Cos_1:y:0rotation_matrix/sub_7:z:0*
T0*#
_output_shapes
:?????????2
rotation_matrix/mul_3
rotation_matrix/addAddV2rotation_matrix/mul_2:z:0rotation_matrix/mul_3:z:0*
T0*#
_output_shapes
:?????????2
rotation_matrix/add
rotation_matrix/sub_8Subrotation_matrix/sub_5:z:0rotation_matrix/add:z:0*
T0*#
_output_shapes
:?????????2
rotation_matrix/sub_8
rotation_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
rotation_matrix/truediv_1/y°
rotation_matrix/truediv_1RealDivrotation_matrix/sub_8:z:0$rotation_matrix/truediv_1/y:output:0*
T0*#
_output_shapes
:?????????2
rotation_matrix/truediv_1r
rotation_matrix/ShapeShapestateful_uniform:z:0*
T0*
_output_shapes
:2
rotation_matrix/Shape
#rotation_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#rotation_matrix/strided_slice/stack
%rotation_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%rotation_matrix/strided_slice/stack_1
%rotation_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%rotation_matrix/strided_slice/stack_2Β
rotation_matrix/strided_sliceStridedSlicerotation_matrix/Shape:output:0,rotation_matrix/strided_slice/stack:output:0.rotation_matrix/strided_slice/stack_1:output:0.rotation_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
rotation_matrix/strided_slicey
rotation_matrix/Cos_2Cosstateful_uniform:z:0*
T0*#
_output_shapes
:?????????2
rotation_matrix/Cos_2
%rotation_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_1/stack£
'rotation_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_1/stack_1£
'rotation_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_1/stack_2χ
rotation_matrix/strided_slice_1StridedSlicerotation_matrix/Cos_2:y:0.rotation_matrix/strided_slice_1/stack:output:00rotation_matrix/strided_slice_1/stack_1:output:00rotation_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_1y
rotation_matrix/Sin_2Sinstateful_uniform:z:0*
T0*#
_output_shapes
:?????????2
rotation_matrix/Sin_2
%rotation_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_2/stack£
'rotation_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_2/stack_1£
'rotation_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_2/stack_2χ
rotation_matrix/strided_slice_2StridedSlicerotation_matrix/Sin_2:y:0.rotation_matrix/strided_slice_2/stack:output:00rotation_matrix/strided_slice_2/stack_1:output:00rotation_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_2
rotation_matrix/NegNeg(rotation_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????2
rotation_matrix/Neg
%rotation_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_3/stack£
'rotation_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_3/stack_1£
'rotation_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_3/stack_2ω
rotation_matrix/strided_slice_3StridedSlicerotation_matrix/truediv:z:0.rotation_matrix/strided_slice_3/stack:output:00rotation_matrix/strided_slice_3/stack_1:output:00rotation_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_3y
rotation_matrix/Sin_3Sinstateful_uniform:z:0*
T0*#
_output_shapes
:?????????2
rotation_matrix/Sin_3
%rotation_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_4/stack£
'rotation_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_4/stack_1£
'rotation_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_4/stack_2χ
rotation_matrix/strided_slice_4StridedSlicerotation_matrix/Sin_3:y:0.rotation_matrix/strided_slice_4/stack:output:00rotation_matrix/strided_slice_4/stack_1:output:00rotation_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_4y
rotation_matrix/Cos_3Cosstateful_uniform:z:0*
T0*#
_output_shapes
:?????????2
rotation_matrix/Cos_3
%rotation_matrix/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_5/stack£
'rotation_matrix/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_5/stack_1£
'rotation_matrix/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_5/stack_2χ
rotation_matrix/strided_slice_5StridedSlicerotation_matrix/Cos_3:y:0.rotation_matrix/strided_slice_5/stack:output:00rotation_matrix/strided_slice_5/stack_1:output:00rotation_matrix/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_5
%rotation_matrix/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_6/stack£
'rotation_matrix/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_6/stack_1£
'rotation_matrix/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_6/stack_2ϋ
rotation_matrix/strided_slice_6StridedSlicerotation_matrix/truediv_1:z:0.rotation_matrix/strided_slice_6/stack:output:00rotation_matrix/strided_slice_6/stack_1:output:00rotation_matrix/strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_6|
rotation_matrix/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
rotation_matrix/zeros/mul/y¬
rotation_matrix/zeros/mulMul&rotation_matrix/strided_slice:output:0$rotation_matrix/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/zeros/mul
rotation_matrix/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :θ2
rotation_matrix/zeros/Less/y§
rotation_matrix/zeros/LessLessrotation_matrix/zeros/mul:z:0%rotation_matrix/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/zeros/Less
rotation_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2 
rotation_matrix/zeros/packed/1Γ
rotation_matrix/zeros/packedPack&rotation_matrix/strided_slice:output:0'rotation_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
rotation_matrix/zeros/packed
rotation_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
rotation_matrix/zeros/Const΅
rotation_matrix/zerosFill%rotation_matrix/zeros/packed:output:0$rotation_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:?????????2
rotation_matrix/zeros|
rotation_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
rotation_matrix/concat/axis¨
rotation_matrix/concatConcatV2(rotation_matrix/strided_slice_1:output:0rotation_matrix/Neg:y:0(rotation_matrix/strided_slice_3:output:0(rotation_matrix/strided_slice_4:output:0(rotation_matrix/strided_slice_5:output:0(rotation_matrix/strided_slice_6:output:0rotation_matrix/zeros:output:0$rotation_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
rotation_matrix/concatX
transform/ShapeShapeinputs*
T0*
_output_shapes
:2
transform/Shape
transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
transform/strided_slice/stack
transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
transform/strided_slice/stack_1
transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
transform/strided_slice/stack_2
transform/strided_sliceStridedSlicetransform/Shape:output:0&transform/strided_slice/stack:output:0(transform/strided_slice/stack_1:output:0(transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
transform/strided_slice¨
$transform/ImageProjectiveTransformV2ImageProjectiveTransformV2inputsrotation_matrix/concat:output:0 transform/strided_slice:output:0*/
_output_shapes
:?????????*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR2&
$transform/ImageProjectiveTransformV2Έ
IdentityIdentity9transform/ImageProjectiveTransformV2:transformed_images:0!^stateful_uniform/StatefulUniform*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????:2D
 stateful_uniform/StatefulUniform stateful_uniform/StatefulUniform:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
°
ο
U__inference_DoubleConv_LessDense_avg_layer_call_and_return_conditional_losses_1716134

inputsX
Ttraining_data_augmentation_random_rotation_stateful_uniform_statefuluniform_resource,
(conv2d_99_conv2d_readvariableop_resource-
)conv2d_99_biasadd_readvariableop_resource-
)conv2d_100_conv2d_readvariableop_resource.
*conv2d_100_biasadd_readvariableop_resource,
(dense_198_matmul_readvariableop_resource-
)dense_198_biasadd_readvariableop_resource,
(dense_199_matmul_readvariableop_resource-
)dense_199_biasadd_readvariableop_resource,
(dense_200_matmul_readvariableop_resource-
)dense_200_biasadd_readvariableop_resource
identity’KTraining_Data_Augmentation/random_rotation/stateful_uniform/StatefulUniform
PTraining_Data_Augmentation/random_flip/random_flip_left_right/control_dependencyIdentityinputs*
T0*
_class
loc:@inputs*/
_output_shapes
:?????????2R
PTraining_Data_Augmentation/random_flip/random_flip_left_right/control_dependency
CTraining_Data_Augmentation/random_flip/random_flip_left_right/ShapeShapeYTraining_Data_Augmentation/random_flip/random_flip_left_right/control_dependency:output:0*
T0*
_output_shapes
:2E
CTraining_Data_Augmentation/random_flip/random_flip_left_right/Shapeπ
QTraining_Data_Augmentation/random_flip/random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2S
QTraining_Data_Augmentation/random_flip/random_flip_left_right/strided_slice/stackτ
STraining_Data_Augmentation/random_flip/random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2U
STraining_Data_Augmentation/random_flip/random_flip_left_right/strided_slice/stack_1τ
STraining_Data_Augmentation/random_flip/random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2U
STraining_Data_Augmentation/random_flip/random_flip_left_right/strided_slice/stack_2Φ
KTraining_Data_Augmentation/random_flip/random_flip_left_right/strided_sliceStridedSliceLTraining_Data_Augmentation/random_flip/random_flip_left_right/Shape:output:0ZTraining_Data_Augmentation/random_flip/random_flip_left_right/strided_slice/stack:output:0\Training_Data_Augmentation/random_flip/random_flip_left_right/strided_slice/stack_1:output:0\Training_Data_Augmentation/random_flip/random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2M
KTraining_Data_Augmentation/random_flip/random_flip_left_right/strided_slice΄
RTraining_Data_Augmentation/random_flip/random_flip_left_right/random_uniform/shapePackTTraining_Data_Augmentation/random_flip/random_flip_left_right/strided_slice:output:0*
N*
T0*
_output_shapes
:2T
RTraining_Data_Augmentation/random_flip/random_flip_left_right/random_uniform/shapeι
PTraining_Data_Augmentation/random_flip/random_flip_left_right/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2R
PTraining_Data_Augmentation/random_flip/random_flip_left_right/random_uniform/minι
PTraining_Data_Augmentation/random_flip/random_flip_left_right/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2R
PTraining_Data_Augmentation/random_flip/random_flip_left_right/random_uniform/maxα
ZTraining_Data_Augmentation/random_flip/random_flip_left_right/random_uniform/RandomUniformRandomUniform[Training_Data_Augmentation/random_flip/random_flip_left_right/random_uniform/shape:output:0*
T0*#
_output_shapes
:?????????*
dtype02\
ZTraining_Data_Augmentation/random_flip/random_flip_left_right/random_uniform/RandomUniform
PTraining_Data_Augmentation/random_flip/random_flip_left_right/random_uniform/MulMulcTraining_Data_Augmentation/random_flip/random_flip_left_right/random_uniform/RandomUniform:output:0YTraining_Data_Augmentation/random_flip/random_flip_left_right/random_uniform/max:output:0*
T0*#
_output_shapes
:?????????2R
PTraining_Data_Augmentation/random_flip/random_flip_left_right/random_uniform/Mulΰ
MTraining_Data_Augmentation/random_flip/random_flip_left_right/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2O
MTraining_Data_Augmentation/random_flip/random_flip_left_right/Reshape/shape/1ΰ
MTraining_Data_Augmentation/random_flip/random_flip_left_right/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2O
MTraining_Data_Augmentation/random_flip/random_flip_left_right/Reshape/shape/2ΰ
MTraining_Data_Augmentation/random_flip/random_flip_left_right/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2O
MTraining_Data_Augmentation/random_flip/random_flip_left_right/Reshape/shape/3?
KTraining_Data_Augmentation/random_flip/random_flip_left_right/Reshape/shapePackTTraining_Data_Augmentation/random_flip/random_flip_left_right/strided_slice:output:0VTraining_Data_Augmentation/random_flip/random_flip_left_right/Reshape/shape/1:output:0VTraining_Data_Augmentation/random_flip/random_flip_left_right/Reshape/shape/2:output:0VTraining_Data_Augmentation/random_flip/random_flip_left_right/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2M
KTraining_Data_Augmentation/random_flip/random_flip_left_right/Reshape/shape?
ETraining_Data_Augmentation/random_flip/random_flip_left_right/ReshapeReshapeTTraining_Data_Augmentation/random_flip/random_flip_left_right/random_uniform/Mul:z:0TTraining_Data_Augmentation/random_flip/random_flip_left_right/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2G
ETraining_Data_Augmentation/random_flip/random_flip_left_right/Reshape
CTraining_Data_Augmentation/random_flip/random_flip_left_right/RoundRoundNTraining_Data_Augmentation/random_flip/random_flip_left_right/Reshape:output:0*
T0*/
_output_shapes
:?????????2E
CTraining_Data_Augmentation/random_flip/random_flip_left_right/Roundζ
LTraining_Data_Augmentation/random_flip/random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:2N
LTraining_Data_Augmentation/random_flip/random_flip_left_right/ReverseV2/axis
GTraining_Data_Augmentation/random_flip/random_flip_left_right/ReverseV2	ReverseV2YTraining_Data_Augmentation/random_flip/random_flip_left_right/control_dependency:output:0UTraining_Data_Augmentation/random_flip/random_flip_left_right/ReverseV2/axis:output:0*
T0*/
_output_shapes
:?????????2I
GTraining_Data_Augmentation/random_flip/random_flip_left_right/ReverseV2β
ATraining_Data_Augmentation/random_flip/random_flip_left_right/mulMulGTraining_Data_Augmentation/random_flip/random_flip_left_right/Round:y:0PTraining_Data_Augmentation/random_flip/random_flip_left_right/ReverseV2:output:0*
T0*/
_output_shapes
:?????????2C
ATraining_Data_Augmentation/random_flip/random_flip_left_right/mulΟ
CTraining_Data_Augmentation/random_flip/random_flip_left_right/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2E
CTraining_Data_Augmentation/random_flip/random_flip_left_right/sub/xή
ATraining_Data_Augmentation/random_flip/random_flip_left_right/subSubLTraining_Data_Augmentation/random_flip/random_flip_left_right/sub/x:output:0GTraining_Data_Augmentation/random_flip/random_flip_left_right/Round:y:0*
T0*/
_output_shapes
:?????????2C
ATraining_Data_Augmentation/random_flip/random_flip_left_right/subν
CTraining_Data_Augmentation/random_flip/random_flip_left_right/mul_1MulETraining_Data_Augmentation/random_flip/random_flip_left_right/sub:z:0YTraining_Data_Augmentation/random_flip/random_flip_left_right/control_dependency:output:0*
T0*/
_output_shapes
:?????????2E
CTraining_Data_Augmentation/random_flip/random_flip_left_right/mul_1Ω
ATraining_Data_Augmentation/random_flip/random_flip_left_right/addAddV2ETraining_Data_Augmentation/random_flip/random_flip_left_right/mul:z:0GTraining_Data_Augmentation/random_flip/random_flip_left_right/mul_1:z:0*
T0*/
_output_shapes
:?????????2C
ATraining_Data_Augmentation/random_flip/random_flip_left_right/addΩ
0Training_Data_Augmentation/random_rotation/ShapeShapeETraining_Data_Augmentation/random_flip/random_flip_left_right/add:z:0*
T0*
_output_shapes
:22
0Training_Data_Augmentation/random_rotation/ShapeΚ
>Training_Data_Augmentation/random_rotation/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2@
>Training_Data_Augmentation/random_rotation/strided_slice/stackΞ
@Training_Data_Augmentation/random_rotation/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2B
@Training_Data_Augmentation/random_rotation/strided_slice/stack_1Ξ
@Training_Data_Augmentation/random_rotation/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2B
@Training_Data_Augmentation/random_rotation/strided_slice/stack_2δ
8Training_Data_Augmentation/random_rotation/strided_sliceStridedSlice9Training_Data_Augmentation/random_rotation/Shape:output:0GTraining_Data_Augmentation/random_rotation/strided_slice/stack:output:0ITraining_Data_Augmentation/random_rotation/strided_slice/stack_1:output:0ITraining_Data_Augmentation/random_rotation/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2:
8Training_Data_Augmentation/random_rotation/strided_sliceΞ
@Training_Data_Augmentation/random_rotation/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2B
@Training_Data_Augmentation/random_rotation/strided_slice_1/stack?
BTraining_Data_Augmentation/random_rotation/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2D
BTraining_Data_Augmentation/random_rotation/strided_slice_1/stack_1?
BTraining_Data_Augmentation/random_rotation/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2D
BTraining_Data_Augmentation/random_rotation/strided_slice_1/stack_2ξ
:Training_Data_Augmentation/random_rotation/strided_slice_1StridedSlice9Training_Data_Augmentation/random_rotation/Shape:output:0ITraining_Data_Augmentation/random_rotation/strided_slice_1/stack:output:0KTraining_Data_Augmentation/random_rotation/strided_slice_1/stack_1:output:0KTraining_Data_Augmentation/random_rotation/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2<
:Training_Data_Augmentation/random_rotation/strided_slice_1ί
/Training_Data_Augmentation/random_rotation/CastCastCTraining_Data_Augmentation/random_rotation/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 21
/Training_Data_Augmentation/random_rotation/CastΞ
@Training_Data_Augmentation/random_rotation/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2B
@Training_Data_Augmentation/random_rotation/strided_slice_2/stack?
BTraining_Data_Augmentation/random_rotation/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2D
BTraining_Data_Augmentation/random_rotation/strided_slice_2/stack_1?
BTraining_Data_Augmentation/random_rotation/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2D
BTraining_Data_Augmentation/random_rotation/strided_slice_2/stack_2ξ
:Training_Data_Augmentation/random_rotation/strided_slice_2StridedSlice9Training_Data_Augmentation/random_rotation/Shape:output:0ITraining_Data_Augmentation/random_rotation/strided_slice_2/stack:output:0KTraining_Data_Augmentation/random_rotation/strided_slice_2/stack_1:output:0KTraining_Data_Augmentation/random_rotation/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2<
:Training_Data_Augmentation/random_rotation/strided_slice_2γ
1Training_Data_Augmentation/random_rotation/Cast_1CastCTraining_Data_Augmentation/random_rotation/strided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: 23
1Training_Data_Augmentation/random_rotation/Cast_1?
ATraining_Data_Augmentation/random_rotation/stateful_uniform/shapePackATraining_Data_Augmentation/random_rotation/strided_slice:output:0*
N*
T0*
_output_shapes
:2C
ATraining_Data_Augmentation/random_rotation/stateful_uniform/shapeΗ
?Training_Data_Augmentation/random_rotation/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *|Ω Ώ2A
?Training_Data_Augmentation/random_rotation/stateful_uniform/minΗ
?Training_Data_Augmentation/random_rotation/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *|Ω ?2A
?Training_Data_Augmentation/random_rotation/stateful_uniform/maxπ
UTraining_Data_Augmentation/random_rotation/stateful_uniform/StatefulUniform/algorithmConst*
_output_shapes
: *
dtype0	*
value	B	 R2W
UTraining_Data_Augmentation/random_rotation/stateful_uniform/StatefulUniform/algorithmη
KTraining_Data_Augmentation/random_rotation/stateful_uniform/StatefulUniformStatefulUniformTtraining_data_augmentation_random_rotation_stateful_uniform_statefuluniform_resource^Training_Data_Augmentation/random_rotation/stateful_uniform/StatefulUniform/algorithm:output:0JTraining_Data_Augmentation/random_rotation/stateful_uniform/shape:output:0*#
_output_shapes
:?????????*
shape_dtype02M
KTraining_Data_Augmentation/random_rotation/stateful_uniform/StatefulUniformΎ
?Training_Data_Augmentation/random_rotation/stateful_uniform/subSubHTraining_Data_Augmentation/random_rotation/stateful_uniform/max:output:0HTraining_Data_Augmentation/random_rotation/stateful_uniform/min:output:0*
T0*
_output_shapes
: 2A
?Training_Data_Augmentation/random_rotation/stateful_uniform/sub?
?Training_Data_Augmentation/random_rotation/stateful_uniform/mulMulTTraining_Data_Augmentation/random_rotation/stateful_uniform/StatefulUniform:output:0CTraining_Data_Augmentation/random_rotation/stateful_uniform/sub:z:0*
T0*#
_output_shapes
:?????????2A
?Training_Data_Augmentation/random_rotation/stateful_uniform/mulΎ
;Training_Data_Augmentation/random_rotation/stateful_uniformAddCTraining_Data_Augmentation/random_rotation/stateful_uniform/mul:z:0HTraining_Data_Augmentation/random_rotation/stateful_uniform/min:output:0*
T0*#
_output_shapes
:?????????2=
;Training_Data_Augmentation/random_rotation/stateful_uniformΙ
@Training_Data_Augmentation/random_rotation/rotation_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2B
@Training_Data_Augmentation/random_rotation/rotation_matrix/sub/yͺ
>Training_Data_Augmentation/random_rotation/rotation_matrix/subSub5Training_Data_Augmentation/random_rotation/Cast_1:y:0ITraining_Data_Augmentation/random_rotation/rotation_matrix/sub/y:output:0*
T0*
_output_shapes
: 2@
>Training_Data_Augmentation/random_rotation/rotation_matrix/subφ
>Training_Data_Augmentation/random_rotation/rotation_matrix/CosCos?Training_Data_Augmentation/random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:?????????2@
>Training_Data_Augmentation/random_rotation/rotation_matrix/CosΝ
BTraining_Data_Augmentation/random_rotation/rotation_matrix/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2D
BTraining_Data_Augmentation/random_rotation/rotation_matrix/sub_1/y°
@Training_Data_Augmentation/random_rotation/rotation_matrix/sub_1Sub5Training_Data_Augmentation/random_rotation/Cast_1:y:0KTraining_Data_Augmentation/random_rotation/rotation_matrix/sub_1/y:output:0*
T0*
_output_shapes
: 2B
@Training_Data_Augmentation/random_rotation/rotation_matrix/sub_1Ώ
>Training_Data_Augmentation/random_rotation/rotation_matrix/mulMulBTraining_Data_Augmentation/random_rotation/rotation_matrix/Cos:y:0DTraining_Data_Augmentation/random_rotation/rotation_matrix/sub_1:z:0*
T0*#
_output_shapes
:?????????2@
>Training_Data_Augmentation/random_rotation/rotation_matrix/mulφ
>Training_Data_Augmentation/random_rotation/rotation_matrix/SinSin?Training_Data_Augmentation/random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:?????????2@
>Training_Data_Augmentation/random_rotation/rotation_matrix/SinΝ
BTraining_Data_Augmentation/random_rotation/rotation_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2D
BTraining_Data_Augmentation/random_rotation/rotation_matrix/sub_2/y?
@Training_Data_Augmentation/random_rotation/rotation_matrix/sub_2Sub3Training_Data_Augmentation/random_rotation/Cast:y:0KTraining_Data_Augmentation/random_rotation/rotation_matrix/sub_2/y:output:0*
T0*
_output_shapes
: 2B
@Training_Data_Augmentation/random_rotation/rotation_matrix/sub_2Γ
@Training_Data_Augmentation/random_rotation/rotation_matrix/mul_1MulBTraining_Data_Augmentation/random_rotation/rotation_matrix/Sin:y:0DTraining_Data_Augmentation/random_rotation/rotation_matrix/sub_2:z:0*
T0*#
_output_shapes
:?????????2B
@Training_Data_Augmentation/random_rotation/rotation_matrix/mul_1Γ
@Training_Data_Augmentation/random_rotation/rotation_matrix/sub_3SubBTraining_Data_Augmentation/random_rotation/rotation_matrix/mul:z:0DTraining_Data_Augmentation/random_rotation/rotation_matrix/mul_1:z:0*
T0*#
_output_shapes
:?????????2B
@Training_Data_Augmentation/random_rotation/rotation_matrix/sub_3Γ
@Training_Data_Augmentation/random_rotation/rotation_matrix/sub_4SubBTraining_Data_Augmentation/random_rotation/rotation_matrix/sub:z:0DTraining_Data_Augmentation/random_rotation/rotation_matrix/sub_3:z:0*
T0*#
_output_shapes
:?????????2B
@Training_Data_Augmentation/random_rotation/rotation_matrix/sub_4Ρ
DTraining_Data_Augmentation/random_rotation/rotation_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2F
DTraining_Data_Augmentation/random_rotation/rotation_matrix/truediv/yΦ
BTraining_Data_Augmentation/random_rotation/rotation_matrix/truedivRealDivDTraining_Data_Augmentation/random_rotation/rotation_matrix/sub_4:z:0MTraining_Data_Augmentation/random_rotation/rotation_matrix/truediv/y:output:0*
T0*#
_output_shapes
:?????????2D
BTraining_Data_Augmentation/random_rotation/rotation_matrix/truedivΝ
BTraining_Data_Augmentation/random_rotation/rotation_matrix/sub_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2D
BTraining_Data_Augmentation/random_rotation/rotation_matrix/sub_5/y?
@Training_Data_Augmentation/random_rotation/rotation_matrix/sub_5Sub3Training_Data_Augmentation/random_rotation/Cast:y:0KTraining_Data_Augmentation/random_rotation/rotation_matrix/sub_5/y:output:0*
T0*
_output_shapes
: 2B
@Training_Data_Augmentation/random_rotation/rotation_matrix/sub_5ϊ
@Training_Data_Augmentation/random_rotation/rotation_matrix/Sin_1Sin?Training_Data_Augmentation/random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:?????????2B
@Training_Data_Augmentation/random_rotation/rotation_matrix/Sin_1Ν
BTraining_Data_Augmentation/random_rotation/rotation_matrix/sub_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2D
BTraining_Data_Augmentation/random_rotation/rotation_matrix/sub_6/y°
@Training_Data_Augmentation/random_rotation/rotation_matrix/sub_6Sub5Training_Data_Augmentation/random_rotation/Cast_1:y:0KTraining_Data_Augmentation/random_rotation/rotation_matrix/sub_6/y:output:0*
T0*
_output_shapes
: 2B
@Training_Data_Augmentation/random_rotation/rotation_matrix/sub_6Ε
@Training_Data_Augmentation/random_rotation/rotation_matrix/mul_2MulDTraining_Data_Augmentation/random_rotation/rotation_matrix/Sin_1:y:0DTraining_Data_Augmentation/random_rotation/rotation_matrix/sub_6:z:0*
T0*#
_output_shapes
:?????????2B
@Training_Data_Augmentation/random_rotation/rotation_matrix/mul_2ϊ
@Training_Data_Augmentation/random_rotation/rotation_matrix/Cos_1Cos?Training_Data_Augmentation/random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:?????????2B
@Training_Data_Augmentation/random_rotation/rotation_matrix/Cos_1Ν
BTraining_Data_Augmentation/random_rotation/rotation_matrix/sub_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2D
BTraining_Data_Augmentation/random_rotation/rotation_matrix/sub_7/y?
@Training_Data_Augmentation/random_rotation/rotation_matrix/sub_7Sub3Training_Data_Augmentation/random_rotation/Cast:y:0KTraining_Data_Augmentation/random_rotation/rotation_matrix/sub_7/y:output:0*
T0*
_output_shapes
: 2B
@Training_Data_Augmentation/random_rotation/rotation_matrix/sub_7Ε
@Training_Data_Augmentation/random_rotation/rotation_matrix/mul_3MulDTraining_Data_Augmentation/random_rotation/rotation_matrix/Cos_1:y:0DTraining_Data_Augmentation/random_rotation/rotation_matrix/sub_7:z:0*
T0*#
_output_shapes
:?????????2B
@Training_Data_Augmentation/random_rotation/rotation_matrix/mul_3Γ
>Training_Data_Augmentation/random_rotation/rotation_matrix/addAddV2DTraining_Data_Augmentation/random_rotation/rotation_matrix/mul_2:z:0DTraining_Data_Augmentation/random_rotation/rotation_matrix/mul_3:z:0*
T0*#
_output_shapes
:?????????2@
>Training_Data_Augmentation/random_rotation/rotation_matrix/addΓ
@Training_Data_Augmentation/random_rotation/rotation_matrix/sub_8SubDTraining_Data_Augmentation/random_rotation/rotation_matrix/sub_5:z:0BTraining_Data_Augmentation/random_rotation/rotation_matrix/add:z:0*
T0*#
_output_shapes
:?????????2B
@Training_Data_Augmentation/random_rotation/rotation_matrix/sub_8Υ
FTraining_Data_Augmentation/random_rotation/rotation_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2H
FTraining_Data_Augmentation/random_rotation/rotation_matrix/truediv_1/yά
DTraining_Data_Augmentation/random_rotation/rotation_matrix/truediv_1RealDivDTraining_Data_Augmentation/random_rotation/rotation_matrix/sub_8:z:0OTraining_Data_Augmentation/random_rotation/rotation_matrix/truediv_1/y:output:0*
T0*#
_output_shapes
:?????????2F
DTraining_Data_Augmentation/random_rotation/rotation_matrix/truediv_1σ
@Training_Data_Augmentation/random_rotation/rotation_matrix/ShapeShape?Training_Data_Augmentation/random_rotation/stateful_uniform:z:0*
T0*
_output_shapes
:2B
@Training_Data_Augmentation/random_rotation/rotation_matrix/Shapeκ
NTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2P
NTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice/stackξ
PTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2R
PTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice/stack_1ξ
PTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2R
PTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice/stack_2Δ
HTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_sliceStridedSliceITraining_Data_Augmentation/random_rotation/rotation_matrix/Shape:output:0WTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice/stack:output:0YTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice/stack_1:output:0YTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2J
HTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_sliceϊ
@Training_Data_Augmentation/random_rotation/rotation_matrix/Cos_2Cos?Training_Data_Augmentation/random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:?????????2B
@Training_Data_Augmentation/random_rotation/rotation_matrix/Cos_2υ
PTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2R
PTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_1/stackω
RTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2T
RTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_1/stack_1ω
RTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2T
RTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_1/stack_2ω
JTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_1StridedSliceDTraining_Data_Augmentation/random_rotation/rotation_matrix/Cos_2:y:0YTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_1/stack:output:0[Training_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_1/stack_1:output:0[Training_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
new_axis_mask2L
JTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_1ϊ
@Training_Data_Augmentation/random_rotation/rotation_matrix/Sin_2Sin?Training_Data_Augmentation/random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:?????????2B
@Training_Data_Augmentation/random_rotation/rotation_matrix/Sin_2υ
PTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2R
PTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_2/stackω
RTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2T
RTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_2/stack_1ω
RTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2T
RTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_2/stack_2ω
JTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_2StridedSliceDTraining_Data_Augmentation/random_rotation/rotation_matrix/Sin_2:y:0YTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_2/stack:output:0[Training_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_2/stack_1:output:0[Training_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
new_axis_mask2L
JTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_2
>Training_Data_Augmentation/random_rotation/rotation_matrix/NegNegSTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????2@
>Training_Data_Augmentation/random_rotation/rotation_matrix/Negυ
PTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        2R
PTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_3/stackω
RTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2T
RTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_3/stack_1ω
RTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2T
RTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_3/stack_2ϋ
JTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_3StridedSliceFTraining_Data_Augmentation/random_rotation/rotation_matrix/truediv:z:0YTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_3/stack:output:0[Training_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_3/stack_1:output:0[Training_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
new_axis_mask2L
JTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_3ϊ
@Training_Data_Augmentation/random_rotation/rotation_matrix/Sin_3Sin?Training_Data_Augmentation/random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:?????????2B
@Training_Data_Augmentation/random_rotation/rotation_matrix/Sin_3υ
PTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        2R
PTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_4/stackω
RTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2T
RTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_4/stack_1ω
RTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2T
RTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_4/stack_2ω
JTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_4StridedSliceDTraining_Data_Augmentation/random_rotation/rotation_matrix/Sin_3:y:0YTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_4/stack:output:0[Training_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_4/stack_1:output:0[Training_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
new_axis_mask2L
JTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_4ϊ
@Training_Data_Augmentation/random_rotation/rotation_matrix/Cos_3Cos?Training_Data_Augmentation/random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:?????????2B
@Training_Data_Augmentation/random_rotation/rotation_matrix/Cos_3υ
PTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        2R
PTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_5/stackω
RTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2T
RTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_5/stack_1ω
RTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2T
RTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_5/stack_2ω
JTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_5StridedSliceDTraining_Data_Augmentation/random_rotation/rotation_matrix/Cos_3:y:0YTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_5/stack:output:0[Training_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_5/stack_1:output:0[Training_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
new_axis_mask2L
JTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_5υ
PTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2R
PTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_6/stackω
RTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2T
RTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_6/stack_1ω
RTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2T
RTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_6/stack_2ύ
JTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_6StridedSliceHTraining_Data_Augmentation/random_rotation/rotation_matrix/truediv_1:z:0YTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_6/stack:output:0[Training_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_6/stack_1:output:0[Training_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
new_axis_mask2L
JTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_6?
FTraining_Data_Augmentation/random_rotation/rotation_matrix/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2H
FTraining_Data_Augmentation/random_rotation/rotation_matrix/zeros/mul/yΨ
DTraining_Data_Augmentation/random_rotation/rotation_matrix/zeros/mulMulQTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice:output:0OTraining_Data_Augmentation/random_rotation/rotation_matrix/zeros/mul/y:output:0*
T0*
_output_shapes
: 2F
DTraining_Data_Augmentation/random_rotation/rotation_matrix/zeros/mulΥ
GTraining_Data_Augmentation/random_rotation/rotation_matrix/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :θ2I
GTraining_Data_Augmentation/random_rotation/rotation_matrix/zeros/Less/yΣ
ETraining_Data_Augmentation/random_rotation/rotation_matrix/zeros/LessLessHTraining_Data_Augmentation/random_rotation/rotation_matrix/zeros/mul:z:0PTraining_Data_Augmentation/random_rotation/rotation_matrix/zeros/Less/y:output:0*
T0*
_output_shapes
: 2G
ETraining_Data_Augmentation/random_rotation/rotation_matrix/zeros/LessΨ
ITraining_Data_Augmentation/random_rotation/rotation_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2K
ITraining_Data_Augmentation/random_rotation/rotation_matrix/zeros/packed/1ο
GTraining_Data_Augmentation/random_rotation/rotation_matrix/zeros/packedPackQTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice:output:0RTraining_Data_Augmentation/random_rotation/rotation_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2I
GTraining_Data_Augmentation/random_rotation/rotation_matrix/zeros/packedΥ
FTraining_Data_Augmentation/random_rotation/rotation_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2H
FTraining_Data_Augmentation/random_rotation/rotation_matrix/zeros/Constα
@Training_Data_Augmentation/random_rotation/rotation_matrix/zerosFillPTraining_Data_Augmentation/random_rotation/rotation_matrix/zeros/packed:output:0OTraining_Data_Augmentation/random_rotation/rotation_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:?????????2B
@Training_Data_Augmentation/random_rotation/rotation_matrix/zeros?
FTraining_Data_Augmentation/random_rotation/rotation_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2H
FTraining_Data_Augmentation/random_rotation/rotation_matrix/concat/axisΦ
ATraining_Data_Augmentation/random_rotation/rotation_matrix/concatConcatV2STraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_1:output:0BTraining_Data_Augmentation/random_rotation/rotation_matrix/Neg:y:0STraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_3:output:0STraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_4:output:0STraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_5:output:0STraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_6:output:0ITraining_Data_Augmentation/random_rotation/rotation_matrix/zeros:output:0OTraining_Data_Augmentation/random_rotation/rotation_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2C
ATraining_Data_Augmentation/random_rotation/rotation_matrix/concatν
:Training_Data_Augmentation/random_rotation/transform/ShapeShapeETraining_Data_Augmentation/random_flip/random_flip_left_right/add:z:0*
T0*
_output_shapes
:2<
:Training_Data_Augmentation/random_rotation/transform/Shapeή
HTraining_Data_Augmentation/random_rotation/transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2J
HTraining_Data_Augmentation/random_rotation/transform/strided_slice/stackβ
JTraining_Data_Augmentation/random_rotation/transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2L
JTraining_Data_Augmentation/random_rotation/transform/strided_slice/stack_1β
JTraining_Data_Augmentation/random_rotation/transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2L
JTraining_Data_Augmentation/random_rotation/transform/strided_slice/stack_2
BTraining_Data_Augmentation/random_rotation/transform/strided_sliceStridedSliceCTraining_Data_Augmentation/random_rotation/transform/Shape:output:0QTraining_Data_Augmentation/random_rotation/transform/strided_slice/stack:output:0STraining_Data_Augmentation/random_rotation/transform/strided_slice/stack_1:output:0STraining_Data_Augmentation/random_rotation/transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2D
BTraining_Data_Augmentation/random_rotation/transform/strided_slice
OTraining_Data_Augmentation/random_rotation/transform/ImageProjectiveTransformV2ImageProjectiveTransformV2ETraining_Data_Augmentation/random_flip/random_flip_left_right/add:z:0JTraining_Data_Augmentation/random_rotation/rotation_matrix/concat:output:0KTraining_Data_Augmentation/random_rotation/transform/strided_slice:output:0*/
_output_shapes
:?????????*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR2Q
OTraining_Data_Augmentation/random_rotation/transform/ImageProjectiveTransformV2Ε
?Training_Data_Augmentation/random_contrast/random_uniform/shapeConst*
_output_shapes
: *
dtype0*
valueB 2A
?Training_Data_Augmentation/random_contrast/random_uniform/shapeΓ
=Training_Data_Augmentation/random_contrast/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2?
=Training_Data_Augmentation/random_contrast/random_uniform/minΓ
=Training_Data_Augmentation/random_contrast/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜ?2?
=Training_Data_Augmentation/random_contrast/random_uniform/max
GTraining_Data_Augmentation/random_contrast/random_uniform/RandomUniformRandomUniformHTraining_Data_Augmentation/random_contrast/random_uniform/shape:output:0*
T0*
_output_shapes
: *
dtype02I
GTraining_Data_Augmentation/random_contrast/random_uniform/RandomUniformΆ
=Training_Data_Augmentation/random_contrast/random_uniform/subSubFTraining_Data_Augmentation/random_contrast/random_uniform/max:output:0FTraining_Data_Augmentation/random_contrast/random_uniform/min:output:0*
T0*
_output_shapes
: 2?
=Training_Data_Augmentation/random_contrast/random_uniform/sub»
=Training_Data_Augmentation/random_contrast/random_uniform/mulMulPTraining_Data_Augmentation/random_contrast/random_uniform/RandomUniform:output:0ATraining_Data_Augmentation/random_contrast/random_uniform/sub:z:0*
T0*
_output_shapes
: 2?
=Training_Data_Augmentation/random_contrast/random_uniform/mul©
9Training_Data_Augmentation/random_contrast/random_uniformAddATraining_Data_Augmentation/random_contrast/random_uniform/mul:z:0FTraining_Data_Augmentation/random_contrast/random_uniform/min:output:0*
T0*
_output_shapes
: 2;
9Training_Data_Augmentation/random_contrast/random_uniformβ
:Training_Data_Augmentation/random_contrast/adjust_contrastAdjustContrastv2dTraining_Data_Augmentation/random_rotation/transform/ImageProjectiveTransformV2:transformed_images:0=Training_Data_Augmentation/random_contrast/random_uniform:z:0*/
_output_shapes
:?????????2<
:Training_Data_Augmentation/random_contrast/adjust_contrast
CTraining_Data_Augmentation/random_contrast/adjust_contrast/IdentityIdentityCTraining_Data_Augmentation/random_contrast/adjust_contrast:output:0*
T0*/
_output_shapes
:?????????2E
CTraining_Data_Augmentation/random_contrast/adjust_contrast/Identity³
conv2d_99/Conv2D/ReadVariableOpReadVariableOp(conv2d_99_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02!
conv2d_99/Conv2D/ReadVariableOp
conv2d_99/Conv2DConv2DLTraining_Data_Augmentation/random_contrast/adjust_contrast/Identity:output:0'conv2d_99/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
conv2d_99/Conv2Dͺ
 conv2d_99/BiasAdd/ReadVariableOpReadVariableOp)conv2d_99_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_99/BiasAdd/ReadVariableOp°
conv2d_99/BiasAddBiasAddconv2d_99/Conv2D:output:0(conv2d_99/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_99/BiasAdd~
conv2d_99/ReluReluconv2d_99/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_99/ReluΫ
average_pooling2d_44/AvgPoolAvgPoolconv2d_99/Relu:activations:0*
T0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2
average_pooling2d_44/AvgPooly
dropout_99/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nΫΆ?2
dropout_99/dropout/Const»
dropout_99/dropout/MulMul%average_pooling2d_44/AvgPool:output:0!dropout_99/dropout/Const:output:0*
T0*/
_output_shapes
:?????????@2
dropout_99/dropout/Mul
dropout_99/dropout/ShapeShape%average_pooling2d_44/AvgPool:output:0*
T0*
_output_shapes
:2
dropout_99/dropout/Shapeέ
/dropout_99/dropout/random_uniform/RandomUniformRandomUniform!dropout_99/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????@*
dtype021
/dropout_99/dropout/random_uniform/RandomUniform
!dropout_99/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2#
!dropout_99/dropout/GreaterEqual/yς
dropout_99/dropout/GreaterEqualGreaterEqual8dropout_99/dropout/random_uniform/RandomUniform:output:0*dropout_99/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????@2!
dropout_99/dropout/GreaterEqual¨
dropout_99/dropout/CastCast#dropout_99/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????@2
dropout_99/dropout/Cast?
dropout_99/dropout/Mul_1Muldropout_99/dropout/Mul:z:0dropout_99/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????@2
dropout_99/dropout/Mul_1Ά
 conv2d_100/Conv2D/ReadVariableOpReadVariableOp)conv2d_100_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02"
 conv2d_100/Conv2D/ReadVariableOpΫ
conv2d_100/Conv2DConv2Ddropout_99/dropout/Mul_1:z:0(conv2d_100/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
conv2d_100/Conv2D­
!conv2d_100/BiasAdd/ReadVariableOpReadVariableOp*conv2d_100_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02#
!conv2d_100/BiasAdd/ReadVariableOp΄
conv2d_100/BiasAddBiasAddconv2d_100/Conv2D:output:0)conv2d_100/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_100/BiasAdd
conv2d_100/ReluReluconv2d_100/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_100/Reluά
average_pooling2d_45/AvgPoolAvgPoolconv2d_100/Relu:activations:0*
T0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2
average_pooling2d_45/AvgPool{
dropout_100/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nΫΆ?2
dropout_100/dropout/ConstΎ
dropout_100/dropout/MulMul%average_pooling2d_45/AvgPool:output:0"dropout_100/dropout/Const:output:0*
T0*/
_output_shapes
:?????????@2
dropout_100/dropout/Mul
dropout_100/dropout/ShapeShape%average_pooling2d_45/AvgPool:output:0*
T0*
_output_shapes
:2
dropout_100/dropout/Shapeΰ
0dropout_100/dropout/random_uniform/RandomUniformRandomUniform"dropout_100/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????@*
dtype022
0dropout_100/dropout/random_uniform/RandomUniform
"dropout_100/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2$
"dropout_100/dropout/GreaterEqual/yφ
 dropout_100/dropout/GreaterEqualGreaterEqual9dropout_100/dropout/random_uniform/RandomUniform:output:0+dropout_100/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????@2"
 dropout_100/dropout/GreaterEqual«
dropout_100/dropout/CastCast$dropout_100/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????@2
dropout_100/dropout/Cast²
dropout_100/dropout/Mul_1Muldropout_100/dropout/Mul:z:0dropout_100/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????@2
dropout_100/dropout/Mul_1u
flatten_66/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 	  2
flatten_66/Const 
flatten_66/ReshapeReshapedropout_100/dropout/Mul_1:z:0flatten_66/Const:output:0*
T0*(
_output_shapes
:?????????2
flatten_66/Reshape­
dense_198/MatMul/ReadVariableOpReadVariableOp(dense_198_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02!
dense_198/MatMul/ReadVariableOp§
dense_198/MatMulMatMulflatten_66/Reshape:output:0'dense_198/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
dense_198/MatMul«
 dense_198/BiasAdd/ReadVariableOpReadVariableOp)dense_198_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 dense_198/BiasAdd/ReadVariableOpͺ
dense_198/BiasAddBiasAdddense_198/MatMul:product:0(dense_198/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
dense_198/BiasAddw
dense_198/ReluReludense_198/BiasAdd:output:0*
T0*(
_output_shapes
:?????????2
dense_198/Relu¬
dense_199/MatMul/ReadVariableOpReadVariableOp(dense_199_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02!
dense_199/MatMul/ReadVariableOp§
dense_199/MatMulMatMuldense_198/Relu:activations:0'dense_199/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_199/MatMulͺ
 dense_199/BiasAdd/ReadVariableOpReadVariableOp)dense_199_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 dense_199/BiasAdd/ReadVariableOp©
dense_199/BiasAddBiasAdddense_199/MatMul:product:0(dense_199/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_199/BiasAddv
dense_199/ReluReludense_199/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_199/Relu«
dense_200/MatMul/ReadVariableOpReadVariableOp(dense_200_matmul_readvariableop_resource*
_output_shapes

:@
*
dtype02!
dense_200/MatMul/ReadVariableOp§
dense_200/MatMulMatMuldense_199/Relu:activations:0'dense_200/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_200/MatMulͺ
 dense_200/BiasAdd/ReadVariableOpReadVariableOp)dense_200_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02"
 dense_200/BiasAdd/ReadVariableOp©
dense_200/BiasAddBiasAdddense_200/MatMul:product:0(dense_200/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_200/BiasAdd
dense_200/SoftmaxSoftmaxdense_200/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
dense_200/Softmax½
IdentityIdentitydense_200/Softmax:softmax:0L^Training_Data_Augmentation/random_rotation/stateful_uniform/StatefulUniform*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:?????????:::::::::::2
KTraining_Data_Augmentation/random_rotation/stateful_uniform/StatefulUniformKTraining_Data_Augmentation/random_rotation/stateful_uniform/StatefulUniform:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
Ε
f
G__inference_dropout_99_layer_call_and_return_conditional_losses_1715573

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nΫΆ?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeΌ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2
dropout/GreaterEqual/yΖ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????@2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????@2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
λ
θ
W__inference_Training_Data_Augmentation_layer_call_and_return_conditional_losses_1716370

inputs=
9random_rotation_stateful_uniform_statefuluniform_resource
identity’0random_rotation/stateful_uniform/StatefulUniformΧ
5random_flip/random_flip_left_right/control_dependencyIdentityinputs*
T0*
_class
loc:@inputs*/
_output_shapes
:?????????27
5random_flip/random_flip_left_right/control_dependencyΒ
(random_flip/random_flip_left_right/ShapeShape>random_flip/random_flip_left_right/control_dependency:output:0*
T0*
_output_shapes
:2*
(random_flip/random_flip_left_right/ShapeΊ
6random_flip/random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 28
6random_flip/random_flip_left_right/strided_slice/stackΎ
8random_flip/random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8random_flip/random_flip_left_right/strided_slice/stack_1Ύ
8random_flip/random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8random_flip/random_flip_left_right/strided_slice/stack_2΄
0random_flip/random_flip_left_right/strided_sliceStridedSlice1random_flip/random_flip_left_right/Shape:output:0?random_flip/random_flip_left_right/strided_slice/stack:output:0Arandom_flip/random_flip_left_right/strided_slice/stack_1:output:0Arandom_flip/random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask22
0random_flip/random_flip_left_right/strided_sliceγ
7random_flip/random_flip_left_right/random_uniform/shapePack9random_flip/random_flip_left_right/strided_slice:output:0*
N*
T0*
_output_shapes
:29
7random_flip/random_flip_left_right/random_uniform/shape³
5random_flip/random_flip_left_right/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    27
5random_flip/random_flip_left_right/random_uniform/min³
5random_flip/random_flip_left_right/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?27
5random_flip/random_flip_left_right/random_uniform/max
?random_flip/random_flip_left_right/random_uniform/RandomUniformRandomUniform@random_flip/random_flip_left_right/random_uniform/shape:output:0*
T0*#
_output_shapes
:?????????*
dtype02A
?random_flip/random_flip_left_right/random_uniform/RandomUniform­
5random_flip/random_flip_left_right/random_uniform/MulMulHrandom_flip/random_flip_left_right/random_uniform/RandomUniform:output:0>random_flip/random_flip_left_right/random_uniform/max:output:0*
T0*#
_output_shapes
:?????????27
5random_flip/random_flip_left_right/random_uniform/Mulͺ
2random_flip/random_flip_left_right/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :24
2random_flip/random_flip_left_right/Reshape/shape/1ͺ
2random_flip/random_flip_left_right/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :24
2random_flip/random_flip_left_right/Reshape/shape/2ͺ
2random_flip/random_flip_left_right/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :24
2random_flip/random_flip_left_right/Reshape/shape/3
0random_flip/random_flip_left_right/Reshape/shapePack9random_flip/random_flip_left_right/strided_slice:output:0;random_flip/random_flip_left_right/Reshape/shape/1:output:0;random_flip/random_flip_left_right/Reshape/shape/2:output:0;random_flip/random_flip_left_right/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:22
0random_flip/random_flip_left_right/Reshape/shape
*random_flip/random_flip_left_right/ReshapeReshape9random_flip/random_flip_left_right/random_uniform/Mul:z:09random_flip/random_flip_left_right/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2,
*random_flip/random_flip_left_right/ReshapeΜ
(random_flip/random_flip_left_right/RoundRound3random_flip/random_flip_left_right/Reshape:output:0*
T0*/
_output_shapes
:?????????2*
(random_flip/random_flip_left_right/Round°
1random_flip/random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:23
1random_flip/random_flip_left_right/ReverseV2/axis
,random_flip/random_flip_left_right/ReverseV2	ReverseV2>random_flip/random_flip_left_right/control_dependency:output:0:random_flip/random_flip_left_right/ReverseV2/axis:output:0*
T0*/
_output_shapes
:?????????2.
,random_flip/random_flip_left_right/ReverseV2φ
&random_flip/random_flip_left_right/mulMul,random_flip/random_flip_left_right/Round:y:05random_flip/random_flip_left_right/ReverseV2:output:0*
T0*/
_output_shapes
:?????????2(
&random_flip/random_flip_left_right/mul
(random_flip/random_flip_left_right/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2*
(random_flip/random_flip_left_right/sub/xς
&random_flip/random_flip_left_right/subSub1random_flip/random_flip_left_right/sub/x:output:0,random_flip/random_flip_left_right/Round:y:0*
T0*/
_output_shapes
:?????????2(
&random_flip/random_flip_left_right/sub
(random_flip/random_flip_left_right/mul_1Mul*random_flip/random_flip_left_right/sub:z:0>random_flip/random_flip_left_right/control_dependency:output:0*
T0*/
_output_shapes
:?????????2*
(random_flip/random_flip_left_right/mul_1ν
&random_flip/random_flip_left_right/addAddV2*random_flip/random_flip_left_right/mul:z:0,random_flip/random_flip_left_right/mul_1:z:0*
T0*/
_output_shapes
:?????????2(
&random_flip/random_flip_left_right/add
random_rotation/ShapeShape*random_flip/random_flip_left_right/add:z:0*
T0*
_output_shapes
:2
random_rotation/Shape
#random_rotation/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#random_rotation/strided_slice/stack
%random_rotation/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%random_rotation/strided_slice/stack_1
%random_rotation/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%random_rotation/strided_slice/stack_2Β
random_rotation/strided_sliceStridedSlicerandom_rotation/Shape:output:0,random_rotation/strided_slice/stack:output:0.random_rotation/strided_slice/stack_1:output:0.random_rotation/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
random_rotation/strided_slice
%random_rotation/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2'
%random_rotation/strided_slice_1/stack
'random_rotation/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'random_rotation/strided_slice_1/stack_1
'random_rotation/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'random_rotation/strided_slice_1/stack_2Μ
random_rotation/strided_slice_1StridedSlicerandom_rotation/Shape:output:0.random_rotation/strided_slice_1/stack:output:00random_rotation/strided_slice_1/stack_1:output:00random_rotation/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
random_rotation/strided_slice_1
random_rotation/CastCast(random_rotation/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
random_rotation/Cast
%random_rotation/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2'
%random_rotation/strided_slice_2/stack
'random_rotation/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'random_rotation/strided_slice_2/stack_1
'random_rotation/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'random_rotation/strided_slice_2/stack_2Μ
random_rotation/strided_slice_2StridedSlicerandom_rotation/Shape:output:0.random_rotation/strided_slice_2/stack:output:00random_rotation/strided_slice_2/stack_1:output:00random_rotation/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
random_rotation/strided_slice_2
random_rotation/Cast_1Cast(random_rotation/strided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
random_rotation/Cast_1?
&random_rotation/stateful_uniform/shapePack&random_rotation/strided_slice:output:0*
N*
T0*
_output_shapes
:2(
&random_rotation/stateful_uniform/shape
$random_rotation/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *|Ω Ώ2&
$random_rotation/stateful_uniform/min
$random_rotation/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *|Ω ?2&
$random_rotation/stateful_uniform/maxΊ
:random_rotation/stateful_uniform/StatefulUniform/algorithmConst*
_output_shapes
: *
dtype0	*
value	B	 R2<
:random_rotation/stateful_uniform/StatefulUniform/algorithmΰ
0random_rotation/stateful_uniform/StatefulUniformStatefulUniform9random_rotation_stateful_uniform_statefuluniform_resourceCrandom_rotation/stateful_uniform/StatefulUniform/algorithm:output:0/random_rotation/stateful_uniform/shape:output:0*#
_output_shapes
:?????????*
shape_dtype022
0random_rotation/stateful_uniform/StatefulUniform?
$random_rotation/stateful_uniform/subSub-random_rotation/stateful_uniform/max:output:0-random_rotation/stateful_uniform/min:output:0*
T0*
_output_shapes
: 2&
$random_rotation/stateful_uniform/subζ
$random_rotation/stateful_uniform/mulMul9random_rotation/stateful_uniform/StatefulUniform:output:0(random_rotation/stateful_uniform/sub:z:0*
T0*#
_output_shapes
:?????????2&
$random_rotation/stateful_uniform/mul?
 random_rotation/stateful_uniformAdd(random_rotation/stateful_uniform/mul:z:0-random_rotation/stateful_uniform/min:output:0*
T0*#
_output_shapes
:?????????2"
 random_rotation/stateful_uniform
%random_rotation/rotation_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2'
%random_rotation/rotation_matrix/sub/yΎ
#random_rotation/rotation_matrix/subSubrandom_rotation/Cast_1:y:0.random_rotation/rotation_matrix/sub/y:output:0*
T0*
_output_shapes
: 2%
#random_rotation/rotation_matrix/sub₯
#random_rotation/rotation_matrix/CosCos$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:?????????2%
#random_rotation/rotation_matrix/Cos
'random_rotation/rotation_matrix/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2)
'random_rotation/rotation_matrix/sub_1/yΔ
%random_rotation/rotation_matrix/sub_1Subrandom_rotation/Cast_1:y:00random_rotation/rotation_matrix/sub_1/y:output:0*
T0*
_output_shapes
: 2'
%random_rotation/rotation_matrix/sub_1Σ
#random_rotation/rotation_matrix/mulMul'random_rotation/rotation_matrix/Cos:y:0)random_rotation/rotation_matrix/sub_1:z:0*
T0*#
_output_shapes
:?????????2%
#random_rotation/rotation_matrix/mul₯
#random_rotation/rotation_matrix/SinSin$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:?????????2%
#random_rotation/rotation_matrix/Sin
'random_rotation/rotation_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2)
'random_rotation/rotation_matrix/sub_2/yΒ
%random_rotation/rotation_matrix/sub_2Subrandom_rotation/Cast:y:00random_rotation/rotation_matrix/sub_2/y:output:0*
T0*
_output_shapes
: 2'
%random_rotation/rotation_matrix/sub_2Χ
%random_rotation/rotation_matrix/mul_1Mul'random_rotation/rotation_matrix/Sin:y:0)random_rotation/rotation_matrix/sub_2:z:0*
T0*#
_output_shapes
:?????????2'
%random_rotation/rotation_matrix/mul_1Χ
%random_rotation/rotation_matrix/sub_3Sub'random_rotation/rotation_matrix/mul:z:0)random_rotation/rotation_matrix/mul_1:z:0*
T0*#
_output_shapes
:?????????2'
%random_rotation/rotation_matrix/sub_3Χ
%random_rotation/rotation_matrix/sub_4Sub'random_rotation/rotation_matrix/sub:z:0)random_rotation/rotation_matrix/sub_3:z:0*
T0*#
_output_shapes
:?????????2'
%random_rotation/rotation_matrix/sub_4
)random_rotation/rotation_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2+
)random_rotation/rotation_matrix/truediv/yκ
'random_rotation/rotation_matrix/truedivRealDiv)random_rotation/rotation_matrix/sub_4:z:02random_rotation/rotation_matrix/truediv/y:output:0*
T0*#
_output_shapes
:?????????2)
'random_rotation/rotation_matrix/truediv
'random_rotation/rotation_matrix/sub_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2)
'random_rotation/rotation_matrix/sub_5/yΒ
%random_rotation/rotation_matrix/sub_5Subrandom_rotation/Cast:y:00random_rotation/rotation_matrix/sub_5/y:output:0*
T0*
_output_shapes
: 2'
%random_rotation/rotation_matrix/sub_5©
%random_rotation/rotation_matrix/Sin_1Sin$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:?????????2'
%random_rotation/rotation_matrix/Sin_1
'random_rotation/rotation_matrix/sub_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2)
'random_rotation/rotation_matrix/sub_6/yΔ
%random_rotation/rotation_matrix/sub_6Subrandom_rotation/Cast_1:y:00random_rotation/rotation_matrix/sub_6/y:output:0*
T0*
_output_shapes
: 2'
%random_rotation/rotation_matrix/sub_6Ω
%random_rotation/rotation_matrix/mul_2Mul)random_rotation/rotation_matrix/Sin_1:y:0)random_rotation/rotation_matrix/sub_6:z:0*
T0*#
_output_shapes
:?????????2'
%random_rotation/rotation_matrix/mul_2©
%random_rotation/rotation_matrix/Cos_1Cos$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:?????????2'
%random_rotation/rotation_matrix/Cos_1
'random_rotation/rotation_matrix/sub_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2)
'random_rotation/rotation_matrix/sub_7/yΒ
%random_rotation/rotation_matrix/sub_7Subrandom_rotation/Cast:y:00random_rotation/rotation_matrix/sub_7/y:output:0*
T0*
_output_shapes
: 2'
%random_rotation/rotation_matrix/sub_7Ω
%random_rotation/rotation_matrix/mul_3Mul)random_rotation/rotation_matrix/Cos_1:y:0)random_rotation/rotation_matrix/sub_7:z:0*
T0*#
_output_shapes
:?????????2'
%random_rotation/rotation_matrix/mul_3Χ
#random_rotation/rotation_matrix/addAddV2)random_rotation/rotation_matrix/mul_2:z:0)random_rotation/rotation_matrix/mul_3:z:0*
T0*#
_output_shapes
:?????????2%
#random_rotation/rotation_matrix/addΧ
%random_rotation/rotation_matrix/sub_8Sub)random_rotation/rotation_matrix/sub_5:z:0'random_rotation/rotation_matrix/add:z:0*
T0*#
_output_shapes
:?????????2'
%random_rotation/rotation_matrix/sub_8
+random_rotation/rotation_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2-
+random_rotation/rotation_matrix/truediv_1/yπ
)random_rotation/rotation_matrix/truediv_1RealDiv)random_rotation/rotation_matrix/sub_8:z:04random_rotation/rotation_matrix/truediv_1/y:output:0*
T0*#
_output_shapes
:?????????2+
)random_rotation/rotation_matrix/truediv_1’
%random_rotation/rotation_matrix/ShapeShape$random_rotation/stateful_uniform:z:0*
T0*
_output_shapes
:2'
%random_rotation/rotation_matrix/Shape΄
3random_rotation/rotation_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3random_rotation/rotation_matrix/strided_slice/stackΈ
5random_rotation/rotation_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5random_rotation/rotation_matrix/strided_slice/stack_1Έ
5random_rotation/rotation_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5random_rotation/rotation_matrix/strided_slice/stack_2’
-random_rotation/rotation_matrix/strided_sliceStridedSlice.random_rotation/rotation_matrix/Shape:output:0<random_rotation/rotation_matrix/strided_slice/stack:output:0>random_rotation/rotation_matrix/strided_slice/stack_1:output:0>random_rotation/rotation_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-random_rotation/rotation_matrix/strided_slice©
%random_rotation/rotation_matrix/Cos_2Cos$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:?????????2'
%random_rotation/rotation_matrix/Cos_2Ώ
5random_rotation/rotation_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        27
5random_rotation/rotation_matrix/strided_slice_1/stackΓ
7random_rotation/rotation_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation/rotation_matrix/strided_slice_1/stack_1Γ
7random_rotation/rotation_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7random_rotation/rotation_matrix/strided_slice_1/stack_2Χ
/random_rotation/rotation_matrix/strided_slice_1StridedSlice)random_rotation/rotation_matrix/Cos_2:y:0>random_rotation/rotation_matrix/strided_slice_1/stack:output:0@random_rotation/rotation_matrix/strided_slice_1/stack_1:output:0@random_rotation/rotation_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
new_axis_mask21
/random_rotation/rotation_matrix/strided_slice_1©
%random_rotation/rotation_matrix/Sin_2Sin$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:?????????2'
%random_rotation/rotation_matrix/Sin_2Ώ
5random_rotation/rotation_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        27
5random_rotation/rotation_matrix/strided_slice_2/stackΓ
7random_rotation/rotation_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation/rotation_matrix/strided_slice_2/stack_1Γ
7random_rotation/rotation_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7random_rotation/rotation_matrix/strided_slice_2/stack_2Χ
/random_rotation/rotation_matrix/strided_slice_2StridedSlice)random_rotation/rotation_matrix/Sin_2:y:0>random_rotation/rotation_matrix/strided_slice_2/stack:output:0@random_rotation/rotation_matrix/strided_slice_2/stack_1:output:0@random_rotation/rotation_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
new_axis_mask21
/random_rotation/rotation_matrix/strided_slice_2½
#random_rotation/rotation_matrix/NegNeg8random_rotation/rotation_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????2%
#random_rotation/rotation_matrix/NegΏ
5random_rotation/rotation_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        27
5random_rotation/rotation_matrix/strided_slice_3/stackΓ
7random_rotation/rotation_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation/rotation_matrix/strided_slice_3/stack_1Γ
7random_rotation/rotation_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7random_rotation/rotation_matrix/strided_slice_3/stack_2Ω
/random_rotation/rotation_matrix/strided_slice_3StridedSlice+random_rotation/rotation_matrix/truediv:z:0>random_rotation/rotation_matrix/strided_slice_3/stack:output:0@random_rotation/rotation_matrix/strided_slice_3/stack_1:output:0@random_rotation/rotation_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
new_axis_mask21
/random_rotation/rotation_matrix/strided_slice_3©
%random_rotation/rotation_matrix/Sin_3Sin$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:?????????2'
%random_rotation/rotation_matrix/Sin_3Ώ
5random_rotation/rotation_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        27
5random_rotation/rotation_matrix/strided_slice_4/stackΓ
7random_rotation/rotation_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation/rotation_matrix/strided_slice_4/stack_1Γ
7random_rotation/rotation_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7random_rotation/rotation_matrix/strided_slice_4/stack_2Χ
/random_rotation/rotation_matrix/strided_slice_4StridedSlice)random_rotation/rotation_matrix/Sin_3:y:0>random_rotation/rotation_matrix/strided_slice_4/stack:output:0@random_rotation/rotation_matrix/strided_slice_4/stack_1:output:0@random_rotation/rotation_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
new_axis_mask21
/random_rotation/rotation_matrix/strided_slice_4©
%random_rotation/rotation_matrix/Cos_3Cos$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:?????????2'
%random_rotation/rotation_matrix/Cos_3Ώ
5random_rotation/rotation_matrix/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        27
5random_rotation/rotation_matrix/strided_slice_5/stackΓ
7random_rotation/rotation_matrix/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation/rotation_matrix/strided_slice_5/stack_1Γ
7random_rotation/rotation_matrix/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7random_rotation/rotation_matrix/strided_slice_5/stack_2Χ
/random_rotation/rotation_matrix/strided_slice_5StridedSlice)random_rotation/rotation_matrix/Cos_3:y:0>random_rotation/rotation_matrix/strided_slice_5/stack:output:0@random_rotation/rotation_matrix/strided_slice_5/stack_1:output:0@random_rotation/rotation_matrix/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
new_axis_mask21
/random_rotation/rotation_matrix/strided_slice_5Ώ
5random_rotation/rotation_matrix/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        27
5random_rotation/rotation_matrix/strided_slice_6/stackΓ
7random_rotation/rotation_matrix/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation/rotation_matrix/strided_slice_6/stack_1Γ
7random_rotation/rotation_matrix/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7random_rotation/rotation_matrix/strided_slice_6/stack_2Ϋ
/random_rotation/rotation_matrix/strided_slice_6StridedSlice-random_rotation/rotation_matrix/truediv_1:z:0>random_rotation/rotation_matrix/strided_slice_6/stack:output:0@random_rotation/rotation_matrix/strided_slice_6/stack_1:output:0@random_rotation/rotation_matrix/strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
new_axis_mask21
/random_rotation/rotation_matrix/strided_slice_6
+random_rotation/rotation_matrix/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2-
+random_rotation/rotation_matrix/zeros/mul/yμ
)random_rotation/rotation_matrix/zeros/mulMul6random_rotation/rotation_matrix/strided_slice:output:04random_rotation/rotation_matrix/zeros/mul/y:output:0*
T0*
_output_shapes
: 2+
)random_rotation/rotation_matrix/zeros/mul
,random_rotation/rotation_matrix/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :θ2.
,random_rotation/rotation_matrix/zeros/Less/yη
*random_rotation/rotation_matrix/zeros/LessLess-random_rotation/rotation_matrix/zeros/mul:z:05random_rotation/rotation_matrix/zeros/Less/y:output:0*
T0*
_output_shapes
: 2,
*random_rotation/rotation_matrix/zeros/Less’
.random_rotation/rotation_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :20
.random_rotation/rotation_matrix/zeros/packed/1
,random_rotation/rotation_matrix/zeros/packedPack6random_rotation/rotation_matrix/strided_slice:output:07random_rotation/rotation_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2.
,random_rotation/rotation_matrix/zeros/packed
+random_rotation/rotation_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+random_rotation/rotation_matrix/zeros/Constυ
%random_rotation/rotation_matrix/zerosFill5random_rotation/rotation_matrix/zeros/packed:output:04random_rotation/rotation_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:?????????2'
%random_rotation/rotation_matrix/zeros
+random_rotation/rotation_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2-
+random_rotation/rotation_matrix/concat/axisΘ
&random_rotation/rotation_matrix/concatConcatV28random_rotation/rotation_matrix/strided_slice_1:output:0'random_rotation/rotation_matrix/Neg:y:08random_rotation/rotation_matrix/strided_slice_3:output:08random_rotation/rotation_matrix/strided_slice_4:output:08random_rotation/rotation_matrix/strided_slice_5:output:08random_rotation/rotation_matrix/strided_slice_6:output:0.random_rotation/rotation_matrix/zeros:output:04random_rotation/rotation_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2(
&random_rotation/rotation_matrix/concat
random_rotation/transform/ShapeShape*random_flip/random_flip_left_right/add:z:0*
T0*
_output_shapes
:2!
random_rotation/transform/Shape¨
-random_rotation/transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2/
-random_rotation/transform/strided_slice/stack¬
/random_rotation/transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/random_rotation/transform/strided_slice/stack_1¬
/random_rotation/transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/random_rotation/transform/strided_slice/stack_2κ
'random_rotation/transform/strided_sliceStridedSlice(random_rotation/transform/Shape:output:06random_rotation/transform/strided_slice/stack:output:08random_rotation/transform/strided_slice/stack_1:output:08random_rotation/transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2)
'random_rotation/transform/strided_slice
4random_rotation/transform/ImageProjectiveTransformV2ImageProjectiveTransformV2*random_flip/random_flip_left_right/add:z:0/random_rotation/rotation_matrix/concat:output:00random_rotation/transform/strided_slice:output:0*/
_output_shapes
:?????????*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR26
4random_rotation/transform/ImageProjectiveTransformV2
$random_contrast/random_uniform/shapeConst*
_output_shapes
: *
dtype0*
valueB 2&
$random_contrast/random_uniform/shape
"random_contrast/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2$
"random_contrast/random_uniform/min
"random_contrast/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜ?2$
"random_contrast/random_uniform/maxΚ
,random_contrast/random_uniform/RandomUniformRandomUniform-random_contrast/random_uniform/shape:output:0*
T0*
_output_shapes
: *
dtype02.
,random_contrast/random_uniform/RandomUniformΚ
"random_contrast/random_uniform/subSub+random_contrast/random_uniform/max:output:0+random_contrast/random_uniform/min:output:0*
T0*
_output_shapes
: 2$
"random_contrast/random_uniform/subΟ
"random_contrast/random_uniform/mulMul5random_contrast/random_uniform/RandomUniform:output:0&random_contrast/random_uniform/sub:z:0*
T0*
_output_shapes
: 2$
"random_contrast/random_uniform/mul½
random_contrast/random_uniformAdd&random_contrast/random_uniform/mul:z:0+random_contrast/random_uniform/min:output:0*
T0*
_output_shapes
: 2 
random_contrast/random_uniformφ
random_contrast/adjust_contrastAdjustContrastv2Irandom_rotation/transform/ImageProjectiveTransformV2:transformed_images:0"random_contrast/random_uniform:z:0*/
_output_shapes
:?????????2!
random_contrast/adjust_contrastΔ
(random_contrast/adjust_contrast/IdentityIdentity(random_contrast/adjust_contrast:output:0*
T0*/
_output_shapes
:?????????2*
(random_contrast/adjust_contrast/Identityΐ
IdentityIdentity1random_contrast/adjust_contrast/Identity:output:01^random_rotation/stateful_uniform/StatefulUniform*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????:2d
0random_rotation/stateful_uniform/StatefulUniform0random_rotation/stateful_uniform/StatefulUniform:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
Ζ
f
-__inference_random_flip_layer_call_fn_1716588

inputs
identity’StatefulPartitionedCallζ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_random_flip_layer_call_and_return_conditional_losses_17152802
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
`
ζ
H__inference_random_flip_layer_call_and_return_conditional_losses_1716645

inputs
identity’9random_flip_left_right/assert_greater_equal/Assert/Assert’@random_flip_left_right/assert_positive/assert_less/Assert/Assertr
random_flip_left_right/ShapeShapeinputs*
T0*
_output_shapes
:2
random_flip_left_right/Shape«
*random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ύ????????2,
*random_flip_left_right/strided_slice/stack¦
,random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,random_flip_left_right/strided_slice/stack_1¦
,random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,random_flip_left_right/strided_slice/stack_2θ
$random_flip_left_right/strided_sliceStridedSlice%random_flip_left_right/Shape:output:03random_flip_left_right/strided_slice/stack:output:05random_flip_left_right/strided_slice/stack_1:output:05random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2&
$random_flip_left_right/strided_slice
,random_flip_left_right/assert_positive/ConstConst*
_output_shapes
: *
dtype0*
value	B : 2.
,random_flip_left_right/assert_positive/Const
7random_flip_left_right/assert_positive/assert_less/LessLess5random_flip_left_right/assert_positive/Const:output:0-random_flip_left_right/strided_slice:output:0*
T0*
_output_shapes
:29
7random_flip_left_right/assert_positive/assert_less/LessΎ
8random_flip_left_right/assert_positive/assert_less/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2:
8random_flip_left_right/assert_positive/assert_less/Const
6random_flip_left_right/assert_positive/assert_less/AllAll;random_flip_left_right/assert_positive/assert_less/Less:z:0Arandom_flip_left_right/assert_positive/assert_less/Const:output:0*
_output_shapes
: 28
6random_flip_left_right/assert_positive/assert_less/Allν
?random_flip_left_right/assert_positive/assert_less/Assert/ConstConst*
_output_shapes
: *
dtype0*;
value2B0 B*inner 3 dims of 'image.shape' must be > 0.2A
?random_flip_left_right/assert_positive/assert_less/Assert/Constύ
Grandom_flip_left_right/assert_positive/assert_less/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*;
value2B0 B*inner 3 dims of 'image.shape' must be > 0.2I
Grandom_flip_left_right/assert_positive/assert_less/Assert/Assert/data_0Γ
@random_flip_left_right/assert_positive/assert_less/Assert/AssertAssert?random_flip_left_right/assert_positive/assert_less/All:output:0Prandom_flip_left_right/assert_positive/assert_less/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 2B
@random_flip_left_right/assert_positive/assert_less/Assert/Assert|
random_flip_left_right/RankConst*
_output_shapes
: *
dtype0*
value	B :2
random_flip_left_right/Rank 
-random_flip_left_right/assert_greater_equal/yConst*
_output_shapes
: *
dtype0*
value	B :2/
-random_flip_left_right/assert_greater_equal/y
8random_flip_left_right/assert_greater_equal/GreaterEqualGreaterEqual$random_flip_left_right/Rank:output:06random_flip_left_right/assert_greater_equal/y:output:0*
T0*
_output_shapes
: 2:
8random_flip_left_right/assert_greater_equal/GreaterEqual©
1random_flip_left_right/assert_greater_equal/ConstConst*
_output_shapes
: *
dtype0*
valueB 23
1random_flip_left_right/assert_greater_equal/Constϋ
/random_flip_left_right/assert_greater_equal/AllAll<random_flip_left_right/assert_greater_equal/GreaterEqual:z:0:random_flip_left_right/assert_greater_equal/Const:output:0*
_output_shapes
: 21
/random_flip_left_right/assert_greater_equal/Allΰ
8random_flip_left_right/assert_greater_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*<
value3B1 B+'image' must be at least three-dimensional.2:
8random_flip_left_right/assert_greater_equal/Assert/Constδ
:random_flip_left_right/assert_greater_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:2<
:random_flip_left_right/assert_greater_equal/Assert/Const_1έ
:random_flip_left_right/assert_greater_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*5
value,B* B$x (random_flip_left_right/Rank:0) = 2<
:random_flip_left_right/assert_greater_equal/Assert/Const_2ο
:random_flip_left_right/assert_greater_equal/Assert/Const_3Const*
_output_shapes
: *
dtype0*G
value>B< B6y (random_flip_left_right/assert_greater_equal/y:0) = 2<
:random_flip_left_right/assert_greater_equal/Assert/Const_3π
@random_flip_left_right/assert_greater_equal/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*<
value3B1 B+'image' must be at least three-dimensional.2B
@random_flip_left_right/assert_greater_equal/Assert/Assert/data_0π
@random_flip_left_right/assert_greater_equal/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:2B
@random_flip_left_right/assert_greater_equal/Assert/Assert/data_1ι
@random_flip_left_right/assert_greater_equal/Assert/Assert/data_2Const*
_output_shapes
: *
dtype0*5
value,B* B$x (random_flip_left_right/Rank:0) = 2B
@random_flip_left_right/assert_greater_equal/Assert/Assert/data_2ϋ
@random_flip_left_right/assert_greater_equal/Assert/Assert/data_4Const*
_output_shapes
: *
dtype0*G
value>B< B6y (random_flip_left_right/assert_greater_equal/y:0) = 2B
@random_flip_left_right/assert_greater_equal/Assert/Assert/data_4?
9random_flip_left_right/assert_greater_equal/Assert/AssertAssert8random_flip_left_right/assert_greater_equal/All:output:0Irandom_flip_left_right/assert_greater_equal/Assert/Assert/data_0:output:0Irandom_flip_left_right/assert_greater_equal/Assert/Assert/data_1:output:0Irandom_flip_left_right/assert_greater_equal/Assert/Assert/data_2:output:0$random_flip_left_right/Rank:output:0Irandom_flip_left_right/assert_greater_equal/Assert/Assert/data_4:output:06random_flip_left_right/assert_greater_equal/y:output:0A^random_flip_left_right/assert_positive/assert_less/Assert/Assert*
T

2*
_output_shapes
 2;
9random_flip_left_right/assert_greater_equal/Assert/AssertΩ
)random_flip_left_right/control_dependencyIdentityinputs:^random_flip_left_right/assert_greater_equal/Assert/AssertA^random_flip_left_right/assert_positive/assert_less/Assert/Assert*
T0*
_class
loc:@inputs*J
_output_shapes8
6:4????????????????????????????????????2+
)random_flip_left_right/control_dependency’
random_flip_left_right/Shape_1Shape2random_flip_left_right/control_dependency:output:0*
T0*
_output_shapes
:2 
random_flip_left_right/Shape_1¦
,random_flip_left_right/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,random_flip_left_right/strided_slice_1/stackͺ
.random_flip_left_right/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.random_flip_left_right/strided_slice_1/stack_1ͺ
.random_flip_left_right/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.random_flip_left_right/strided_slice_1/stack_2ψ
&random_flip_left_right/strided_slice_1StridedSlice'random_flip_left_right/Shape_1:output:05random_flip_left_right/strided_slice_1/stack:output:07random_flip_left_right/strided_slice_1/stack_1:output:07random_flip_left_right/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&random_flip_left_right/strided_slice_1Α
+random_flip_left_right/random_uniform/shapePack/random_flip_left_right/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2-
+random_flip_left_right/random_uniform/shape
)random_flip_left_right/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2+
)random_flip_left_right/random_uniform/min
)random_flip_left_right/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2+
)random_flip_left_right/random_uniform/maxμ
3random_flip_left_right/random_uniform/RandomUniformRandomUniform4random_flip_left_right/random_uniform/shape:output:0*
T0*#
_output_shapes
:?????????*
dtype025
3random_flip_left_right/random_uniform/RandomUniformύ
)random_flip_left_right/random_uniform/MulMul<random_flip_left_right/random_uniform/RandomUniform:output:02random_flip_left_right/random_uniform/max:output:0*
T0*#
_output_shapes
:?????????2+
)random_flip_left_right/random_uniform/Mul
&random_flip_left_right/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2(
&random_flip_left_right/Reshape/shape/1
&random_flip_left_right/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2(
&random_flip_left_right/Reshape/shape/2
&random_flip_left_right/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2(
&random_flip_left_right/Reshape/shape/3Ζ
$random_flip_left_right/Reshape/shapePack/random_flip_left_right/strided_slice_1:output:0/random_flip_left_right/Reshape/shape/1:output:0/random_flip_left_right/Reshape/shape/2:output:0/random_flip_left_right/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2&
$random_flip_left_right/Reshape/shapeγ
random_flip_left_right/ReshapeReshape-random_flip_left_right/random_uniform/Mul:z:0-random_flip_left_right/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2 
random_flip_left_right/Reshape¨
random_flip_left_right/RoundRound'random_flip_left_right/Reshape:output:0*
T0*/
_output_shapes
:?????????2
random_flip_left_right/Round
%random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:2'
%random_flip_left_right/ReverseV2/axis
 random_flip_left_right/ReverseV2	ReverseV22random_flip_left_right/control_dependency:output:0.random_flip_left_right/ReverseV2/axis:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2"
 random_flip_left_right/ReverseV2α
random_flip_left_right/mulMul random_flip_left_right/Round:y:0)random_flip_left_right/ReverseV2:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
random_flip_left_right/mul
random_flip_left_right/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
random_flip_left_right/sub/xΒ
random_flip_left_right/subSub%random_flip_left_right/sub/x:output:0 random_flip_left_right/Round:y:0*
T0*/
_output_shapes
:?????????2
random_flip_left_right/subμ
random_flip_left_right/mul_1Mulrandom_flip_left_right/sub:z:02random_flip_left_right/control_dependency:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
random_flip_left_right/mul_1Ψ
random_flip_left_right/addAddV2random_flip_left_right/mul:z:0 random_flip_left_right/mul_1:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
random_flip_left_right/add
IdentityIdentityrandom_flip_left_right/add:z:0:^random_flip_left_right/assert_greater_equal/Assert/AssertA^random_flip_left_right/assert_positive/assert_less/Assert/Assert*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????2v
9random_flip_left_right/assert_greater_equal/Assert/Assert9random_flip_left_right/assert_greater_equal/Assert/Assert2
@random_flip_left_right/assert_positive/assert_less/Assert/Assert@random_flip_left_right/assert_positive/assert_less/Assert/Assert:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
Ή
R
6__inference_average_pooling2d_45_layer_call_fn_1715514

inputs
identityς
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_average_pooling2d_45_layer_call_and_return_conditional_losses_17155082
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
`
ζ
H__inference_random_flip_layer_call_and_return_conditional_losses_1715237

inputs
identity’9random_flip_left_right/assert_greater_equal/Assert/Assert’@random_flip_left_right/assert_positive/assert_less/Assert/Assertr
random_flip_left_right/ShapeShapeinputs*
T0*
_output_shapes
:2
random_flip_left_right/Shape«
*random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ύ????????2,
*random_flip_left_right/strided_slice/stack¦
,random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,random_flip_left_right/strided_slice/stack_1¦
,random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,random_flip_left_right/strided_slice/stack_2θ
$random_flip_left_right/strided_sliceStridedSlice%random_flip_left_right/Shape:output:03random_flip_left_right/strided_slice/stack:output:05random_flip_left_right/strided_slice/stack_1:output:05random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2&
$random_flip_left_right/strided_slice
,random_flip_left_right/assert_positive/ConstConst*
_output_shapes
: *
dtype0*
value	B : 2.
,random_flip_left_right/assert_positive/Const
7random_flip_left_right/assert_positive/assert_less/LessLess5random_flip_left_right/assert_positive/Const:output:0-random_flip_left_right/strided_slice:output:0*
T0*
_output_shapes
:29
7random_flip_left_right/assert_positive/assert_less/LessΎ
8random_flip_left_right/assert_positive/assert_less/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2:
8random_flip_left_right/assert_positive/assert_less/Const
6random_flip_left_right/assert_positive/assert_less/AllAll;random_flip_left_right/assert_positive/assert_less/Less:z:0Arandom_flip_left_right/assert_positive/assert_less/Const:output:0*
_output_shapes
: 28
6random_flip_left_right/assert_positive/assert_less/Allν
?random_flip_left_right/assert_positive/assert_less/Assert/ConstConst*
_output_shapes
: *
dtype0*;
value2B0 B*inner 3 dims of 'image.shape' must be > 0.2A
?random_flip_left_right/assert_positive/assert_less/Assert/Constύ
Grandom_flip_left_right/assert_positive/assert_less/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*;
value2B0 B*inner 3 dims of 'image.shape' must be > 0.2I
Grandom_flip_left_right/assert_positive/assert_less/Assert/Assert/data_0Γ
@random_flip_left_right/assert_positive/assert_less/Assert/AssertAssert?random_flip_left_right/assert_positive/assert_less/All:output:0Prandom_flip_left_right/assert_positive/assert_less/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 2B
@random_flip_left_right/assert_positive/assert_less/Assert/Assert|
random_flip_left_right/RankConst*
_output_shapes
: *
dtype0*
value	B :2
random_flip_left_right/Rank 
-random_flip_left_right/assert_greater_equal/yConst*
_output_shapes
: *
dtype0*
value	B :2/
-random_flip_left_right/assert_greater_equal/y
8random_flip_left_right/assert_greater_equal/GreaterEqualGreaterEqual$random_flip_left_right/Rank:output:06random_flip_left_right/assert_greater_equal/y:output:0*
T0*
_output_shapes
: 2:
8random_flip_left_right/assert_greater_equal/GreaterEqual©
1random_flip_left_right/assert_greater_equal/ConstConst*
_output_shapes
: *
dtype0*
valueB 23
1random_flip_left_right/assert_greater_equal/Constϋ
/random_flip_left_right/assert_greater_equal/AllAll<random_flip_left_right/assert_greater_equal/GreaterEqual:z:0:random_flip_left_right/assert_greater_equal/Const:output:0*
_output_shapes
: 21
/random_flip_left_right/assert_greater_equal/Allΰ
8random_flip_left_right/assert_greater_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*<
value3B1 B+'image' must be at least three-dimensional.2:
8random_flip_left_right/assert_greater_equal/Assert/Constδ
:random_flip_left_right/assert_greater_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:2<
:random_flip_left_right/assert_greater_equal/Assert/Const_1έ
:random_flip_left_right/assert_greater_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*5
value,B* B$x (random_flip_left_right/Rank:0) = 2<
:random_flip_left_right/assert_greater_equal/Assert/Const_2ο
:random_flip_left_right/assert_greater_equal/Assert/Const_3Const*
_output_shapes
: *
dtype0*G
value>B< B6y (random_flip_left_right/assert_greater_equal/y:0) = 2<
:random_flip_left_right/assert_greater_equal/Assert/Const_3π
@random_flip_left_right/assert_greater_equal/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*<
value3B1 B+'image' must be at least three-dimensional.2B
@random_flip_left_right/assert_greater_equal/Assert/Assert/data_0π
@random_flip_left_right/assert_greater_equal/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:2B
@random_flip_left_right/assert_greater_equal/Assert/Assert/data_1ι
@random_flip_left_right/assert_greater_equal/Assert/Assert/data_2Const*
_output_shapes
: *
dtype0*5
value,B* B$x (random_flip_left_right/Rank:0) = 2B
@random_flip_left_right/assert_greater_equal/Assert/Assert/data_2ϋ
@random_flip_left_right/assert_greater_equal/Assert/Assert/data_4Const*
_output_shapes
: *
dtype0*G
value>B< B6y (random_flip_left_right/assert_greater_equal/y:0) = 2B
@random_flip_left_right/assert_greater_equal/Assert/Assert/data_4?
9random_flip_left_right/assert_greater_equal/Assert/AssertAssert8random_flip_left_right/assert_greater_equal/All:output:0Irandom_flip_left_right/assert_greater_equal/Assert/Assert/data_0:output:0Irandom_flip_left_right/assert_greater_equal/Assert/Assert/data_1:output:0Irandom_flip_left_right/assert_greater_equal/Assert/Assert/data_2:output:0$random_flip_left_right/Rank:output:0Irandom_flip_left_right/assert_greater_equal/Assert/Assert/data_4:output:06random_flip_left_right/assert_greater_equal/y:output:0A^random_flip_left_right/assert_positive/assert_less/Assert/Assert*
T

2*
_output_shapes
 2;
9random_flip_left_right/assert_greater_equal/Assert/AssertΩ
)random_flip_left_right/control_dependencyIdentityinputs:^random_flip_left_right/assert_greater_equal/Assert/AssertA^random_flip_left_right/assert_positive/assert_less/Assert/Assert*
T0*
_class
loc:@inputs*J
_output_shapes8
6:4????????????????????????????????????2+
)random_flip_left_right/control_dependency’
random_flip_left_right/Shape_1Shape2random_flip_left_right/control_dependency:output:0*
T0*
_output_shapes
:2 
random_flip_left_right/Shape_1¦
,random_flip_left_right/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,random_flip_left_right/strided_slice_1/stackͺ
.random_flip_left_right/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.random_flip_left_right/strided_slice_1/stack_1ͺ
.random_flip_left_right/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.random_flip_left_right/strided_slice_1/stack_2ψ
&random_flip_left_right/strided_slice_1StridedSlice'random_flip_left_right/Shape_1:output:05random_flip_left_right/strided_slice_1/stack:output:07random_flip_left_right/strided_slice_1/stack_1:output:07random_flip_left_right/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&random_flip_left_right/strided_slice_1Α
+random_flip_left_right/random_uniform/shapePack/random_flip_left_right/strided_slice_1:output:0*
N*
T0*
_output_shapes
:2-
+random_flip_left_right/random_uniform/shape
)random_flip_left_right/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2+
)random_flip_left_right/random_uniform/min
)random_flip_left_right/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2+
)random_flip_left_right/random_uniform/maxμ
3random_flip_left_right/random_uniform/RandomUniformRandomUniform4random_flip_left_right/random_uniform/shape:output:0*
T0*#
_output_shapes
:?????????*
dtype025
3random_flip_left_right/random_uniform/RandomUniformύ
)random_flip_left_right/random_uniform/MulMul<random_flip_left_right/random_uniform/RandomUniform:output:02random_flip_left_right/random_uniform/max:output:0*
T0*#
_output_shapes
:?????????2+
)random_flip_left_right/random_uniform/Mul
&random_flip_left_right/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2(
&random_flip_left_right/Reshape/shape/1
&random_flip_left_right/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2(
&random_flip_left_right/Reshape/shape/2
&random_flip_left_right/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2(
&random_flip_left_right/Reshape/shape/3Ζ
$random_flip_left_right/Reshape/shapePack/random_flip_left_right/strided_slice_1:output:0/random_flip_left_right/Reshape/shape/1:output:0/random_flip_left_right/Reshape/shape/2:output:0/random_flip_left_right/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2&
$random_flip_left_right/Reshape/shapeγ
random_flip_left_right/ReshapeReshape-random_flip_left_right/random_uniform/Mul:z:0-random_flip_left_right/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2 
random_flip_left_right/Reshape¨
random_flip_left_right/RoundRound'random_flip_left_right/Reshape:output:0*
T0*/
_output_shapes
:?????????2
random_flip_left_right/Round
%random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:2'
%random_flip_left_right/ReverseV2/axis
 random_flip_left_right/ReverseV2	ReverseV22random_flip_left_right/control_dependency:output:0.random_flip_left_right/ReverseV2/axis:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2"
 random_flip_left_right/ReverseV2α
random_flip_left_right/mulMul random_flip_left_right/Round:y:0)random_flip_left_right/ReverseV2:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
random_flip_left_right/mul
random_flip_left_right/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
random_flip_left_right/sub/xΒ
random_flip_left_right/subSub%random_flip_left_right/sub/x:output:0 random_flip_left_right/Round:y:0*
T0*/
_output_shapes
:?????????2
random_flip_left_right/subμ
random_flip_left_right/mul_1Mulrandom_flip_left_right/sub:z:02random_flip_left_right/control_dependency:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
random_flip_left_right/mul_1Ψ
random_flip_left_right/addAddV2random_flip_left_right/mul:z:0 random_flip_left_right/mul_1:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
random_flip_left_right/add
IdentityIdentityrandom_flip_left_right/add:z:0:^random_flip_left_right/assert_greater_equal/Assert/AssertA^random_flip_left_right/assert_positive/assert_less/Assert/Assert*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????2v
9random_flip_left_right/assert_greater_equal/Assert/Assert9random_flip_left_right/assert_greater_equal/Assert/Assert2
@random_flip_left_right/assert_positive/assert_less/Assert/Assert@random_flip_left_right/assert_positive/assert_less/Assert/Assert:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs

m
Q__inference_average_pooling2d_45_layer_call_and_return_conditional_losses_1715508

inputs
identityΆ
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
AvgPool
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
	
―
G__inference_conv2d_100_layer_call_and_return_conditional_losses_1716444

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp€
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@:::W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
‘

<__inference_Training_Data_Augmentation_layer_call_fn_1715478
random_flip_input
unknown
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallrandom_flip_inputunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *`
f[RY
W__inference_Training_Data_Augmentation_layer_call_and_return_conditional_losses_17154732
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????:22
StatefulPartitionedCallStatefulPartitionedCall:b ^
/
_output_shapes
:?????????
+
_user_specified_namerandom_flip_input
ε/

U__inference_DoubleConv_LessDense_avg_layer_call_and_return_conditional_losses_1715882

inputs
conv2d_99_1715851
conv2d_99_1715853
conv2d_100_1715858
conv2d_100_1715860
dense_198_1715866
dense_198_1715868
dense_199_1715871
dense_199_1715873
dense_200_1715876
dense_200_1715878
identity’"conv2d_100/StatefulPartitionedCall’!conv2d_99/StatefulPartitionedCall’!dense_198/StatefulPartitionedCall’!dense_199/StatefulPartitionedCall’!dense_200/StatefulPartitionedCall
*Training_Data_Augmentation/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *`
f[RY
W__inference_Training_Data_Augmentation_layer_call_and_return_conditional_losses_17154872,
*Training_Data_Augmentation/PartitionedCallΡ
!conv2d_99/StatefulPartitionedCallStatefulPartitionedCall3Training_Data_Augmentation/PartitionedCall:output:0conv2d_99_1715851conv2d_99_1715853*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_99_layer_call_and_return_conditional_losses_17155442#
!conv2d_99/StatefulPartitionedCall₯
$average_pooling2d_44/PartitionedCallPartitionedCall*conv2d_99/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_average_pooling2d_44_layer_call_and_return_conditional_losses_17154962&
$average_pooling2d_44/PartitionedCall
dropout_99/PartitionedCallPartitionedCall-average_pooling2d_44/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_99_layer_call_and_return_conditional_losses_17155782
dropout_99/PartitionedCallΖ
"conv2d_100/StatefulPartitionedCallStatefulPartitionedCall#dropout_99/PartitionedCall:output:0conv2d_100_1715858conv2d_100_1715860*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_conv2d_100_layer_call_and_return_conditional_losses_17156022$
"conv2d_100/StatefulPartitionedCall¦
$average_pooling2d_45/PartitionedCallPartitionedCall+conv2d_100/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_average_pooling2d_45_layer_call_and_return_conditional_losses_17155082&
$average_pooling2d_45/PartitionedCall
dropout_100/PartitionedCallPartitionedCall-average_pooling2d_45/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_100_layer_call_and_return_conditional_losses_17156362
dropout_100/PartitionedCallϊ
flatten_66/PartitionedCallPartitionedCall$dropout_100/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_66_layer_call_and_return_conditional_losses_17156552
flatten_66/PartitionedCallΊ
!dense_198/StatefulPartitionedCallStatefulPartitionedCall#flatten_66/PartitionedCall:output:0dense_198_1715866dense_198_1715868*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_198_layer_call_and_return_conditional_losses_17156742#
!dense_198/StatefulPartitionedCallΐ
!dense_199/StatefulPartitionedCallStatefulPartitionedCall*dense_198/StatefulPartitionedCall:output:0dense_199_1715871dense_199_1715873*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_199_layer_call_and_return_conditional_losses_17157012#
!dense_199/StatefulPartitionedCallΐ
!dense_200/StatefulPartitionedCallStatefulPartitionedCall*dense_199/StatefulPartitionedCall:output:0dense_200_1715876dense_200_1715878*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_200_layer_call_and_return_conditional_losses_17157282#
!dense_200/StatefulPartitionedCall³
IdentityIdentity*dense_200/StatefulPartitionedCall:output:0#^conv2d_100/StatefulPartitionedCall"^conv2d_99/StatefulPartitionedCall"^dense_198/StatefulPartitionedCall"^dense_199/StatefulPartitionedCall"^dense_200/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????::::::::::2H
"conv2d_100/StatefulPartitionedCall"conv2d_100/StatefulPartitionedCall2F
!conv2d_99/StatefulPartitionedCall!conv2d_99/StatefulPartitionedCall2F
!dense_198/StatefulPartitionedCall!dense_198/StatefulPartitionedCall2F
!dense_199/StatefulPartitionedCall!dense_199/StatefulPartitionedCall2F
!dense_200/StatefulPartitionedCall!dense_200/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?/
Η
U__inference_DoubleConv_LessDense_avg_layer_call_and_return_conditional_losses_1716179

inputs,
(conv2d_99_conv2d_readvariableop_resource-
)conv2d_99_biasadd_readvariableop_resource-
)conv2d_100_conv2d_readvariableop_resource.
*conv2d_100_biasadd_readvariableop_resource,
(dense_198_matmul_readvariableop_resource-
)dense_198_biasadd_readvariableop_resource,
(dense_199_matmul_readvariableop_resource-
)dense_199_biasadd_readvariableop_resource,
(dense_200_matmul_readvariableop_resource-
)dense_200_biasadd_readvariableop_resource
identity³
conv2d_99/Conv2D/ReadVariableOpReadVariableOp(conv2d_99_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02!
conv2d_99/Conv2D/ReadVariableOpΒ
conv2d_99/Conv2DConv2Dinputs'conv2d_99/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
conv2d_99/Conv2Dͺ
 conv2d_99/BiasAdd/ReadVariableOpReadVariableOp)conv2d_99_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_99/BiasAdd/ReadVariableOp°
conv2d_99/BiasAddBiasAddconv2d_99/Conv2D:output:0(conv2d_99/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_99/BiasAdd~
conv2d_99/ReluReluconv2d_99/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_99/ReluΫ
average_pooling2d_44/AvgPoolAvgPoolconv2d_99/Relu:activations:0*
T0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2
average_pooling2d_44/AvgPool
dropout_99/IdentityIdentity%average_pooling2d_44/AvgPool:output:0*
T0*/
_output_shapes
:?????????@2
dropout_99/IdentityΆ
 conv2d_100/Conv2D/ReadVariableOpReadVariableOp)conv2d_100_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02"
 conv2d_100/Conv2D/ReadVariableOpΫ
conv2d_100/Conv2DConv2Ddropout_99/Identity:output:0(conv2d_100/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
conv2d_100/Conv2D­
!conv2d_100/BiasAdd/ReadVariableOpReadVariableOp*conv2d_100_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02#
!conv2d_100/BiasAdd/ReadVariableOp΄
conv2d_100/BiasAddBiasAddconv2d_100/Conv2D:output:0)conv2d_100/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_100/BiasAdd
conv2d_100/ReluReluconv2d_100/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_100/Reluά
average_pooling2d_45/AvgPoolAvgPoolconv2d_100/Relu:activations:0*
T0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2
average_pooling2d_45/AvgPool
dropout_100/IdentityIdentity%average_pooling2d_45/AvgPool:output:0*
T0*/
_output_shapes
:?????????@2
dropout_100/Identityu
flatten_66/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 	  2
flatten_66/Const 
flatten_66/ReshapeReshapedropout_100/Identity:output:0flatten_66/Const:output:0*
T0*(
_output_shapes
:?????????2
flatten_66/Reshape­
dense_198/MatMul/ReadVariableOpReadVariableOp(dense_198_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02!
dense_198/MatMul/ReadVariableOp§
dense_198/MatMulMatMulflatten_66/Reshape:output:0'dense_198/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
dense_198/MatMul«
 dense_198/BiasAdd/ReadVariableOpReadVariableOp)dense_198_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 dense_198/BiasAdd/ReadVariableOpͺ
dense_198/BiasAddBiasAdddense_198/MatMul:product:0(dense_198/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
dense_198/BiasAddw
dense_198/ReluReludense_198/BiasAdd:output:0*
T0*(
_output_shapes
:?????????2
dense_198/Relu¬
dense_199/MatMul/ReadVariableOpReadVariableOp(dense_199_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02!
dense_199/MatMul/ReadVariableOp§
dense_199/MatMulMatMuldense_198/Relu:activations:0'dense_199/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_199/MatMulͺ
 dense_199/BiasAdd/ReadVariableOpReadVariableOp)dense_199_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 dense_199/BiasAdd/ReadVariableOp©
dense_199/BiasAddBiasAdddense_199/MatMul:product:0(dense_199/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_199/BiasAddv
dense_199/ReluReludense_199/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_199/Relu«
dense_200/MatMul/ReadVariableOpReadVariableOp(dense_200_matmul_readvariableop_resource*
_output_shapes

:@
*
dtype02!
dense_200/MatMul/ReadVariableOp§
dense_200/MatMulMatMuldense_199/Relu:activations:0'dense_200/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_200/MatMulͺ
 dense_200/BiasAdd/ReadVariableOpReadVariableOp)dense_200_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02"
 dense_200/BiasAdd/ReadVariableOp©
dense_200/BiasAddBiasAdddense_200/MatMul:product:0(dense_200/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_200/BiasAdd
dense_200/SoftmaxSoftmaxdense_200/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
dense_200/Softmaxo
IdentityIdentitydense_200/Softmax:softmax:0*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????:::::::::::W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs


+__inference_conv2d_99_layer_call_fn_1716406

inputs
unknown
	unknown_0
identity’StatefulPartitionedCallώ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_99_layer_call_and_return_conditional_losses_17155442
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
ό	
°
:__inference_DoubleConv_LessDense_avg_layer_call_fn_1715845$
 training_data_augmentation_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCall training_data_augmentation_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *^
fYRW
U__inference_DoubleConv_LessDense_avg_layer_call_and_return_conditional_losses_17158202
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:?????????:::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:q m
/
_output_shapes
:?????????
:
_user_specified_name" Training_Data_Augmentation_input
?
?
F__inference_dense_199_layer_call_and_return_conditional_losses_1715701

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:::P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
Έ
H
,__inference_dropout_99_layer_call_fn_1716433

inputs
identityΝ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_99_layer_call_and_return_conditional_losses_17155782
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
Γ
d
H__inference_random_flip_layer_call_and_return_conditional_losses_1716649

inputs
identity}
IdentityIdentityinputs*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
·
s
W__inference_Training_Data_Augmentation_layer_call_and_return_conditional_losses_1715487

inputs
identityζ
random_flip/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_random_flip_layer_call_and_return_conditional_losses_17152842
random_flip/PartitionedCall
random_rotation/PartitionedCallPartitionedCall$random_flip/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_random_rotation_layer_call_and_return_conditional_losses_17154062!
random_rotation/PartitionedCall
random_contrast/PartitionedCallPartitionedCall(random_rotation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_random_contrast_layer_call_and_return_conditional_losses_17154402!
random_contrast/PartitionedCall
IdentityIdentity(random_contrast/PartitionedCall:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
6
Σ
U__inference_DoubleConv_LessDense_avg_layer_call_and_return_conditional_losses_1715745$
 training_data_augmentation_input&
"training_data_augmentation_1715530
conv2d_99_1715555
conv2d_99_1715557
conv2d_100_1715613
conv2d_100_1715615
dense_198_1715685
dense_198_1715687
dense_199_1715712
dense_199_1715714
dense_200_1715739
dense_200_1715741
identity’2Training_Data_Augmentation/StatefulPartitionedCall’"conv2d_100/StatefulPartitionedCall’!conv2d_99/StatefulPartitionedCall’!dense_198/StatefulPartitionedCall’!dense_199/StatefulPartitionedCall’!dense_200/StatefulPartitionedCall’#dropout_100/StatefulPartitionedCall’"dropout_99/StatefulPartitionedCallκ
2Training_Data_Augmentation/StatefulPartitionedCallStatefulPartitionedCall training_data_augmentation_input"training_data_augmentation_1715530*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *`
f[RY
W__inference_Training_Data_Augmentation_layer_call_and_return_conditional_losses_171547324
2Training_Data_Augmentation/StatefulPartitionedCallΩ
!conv2d_99/StatefulPartitionedCallStatefulPartitionedCall;Training_Data_Augmentation/StatefulPartitionedCall:output:0conv2d_99_1715555conv2d_99_1715557*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_99_layer_call_and_return_conditional_losses_17155442#
!conv2d_99/StatefulPartitionedCall₯
$average_pooling2d_44/PartitionedCallPartitionedCall*conv2d_99/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_average_pooling2d_44_layer_call_and_return_conditional_losses_17154962&
$average_pooling2d_44/PartitionedCall’
"dropout_99/StatefulPartitionedCallStatefulPartitionedCall-average_pooling2d_44/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_99_layer_call_and_return_conditional_losses_17155732$
"dropout_99/StatefulPartitionedCallΞ
"conv2d_100/StatefulPartitionedCallStatefulPartitionedCall+dropout_99/StatefulPartitionedCall:output:0conv2d_100_1715613conv2d_100_1715615*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_conv2d_100_layer_call_and_return_conditional_losses_17156022$
"conv2d_100/StatefulPartitionedCall¦
$average_pooling2d_45/PartitionedCallPartitionedCall+conv2d_100/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_average_pooling2d_45_layer_call_and_return_conditional_losses_17155082&
$average_pooling2d_45/PartitionedCallΚ
#dropout_100/StatefulPartitionedCallStatefulPartitionedCall-average_pooling2d_45/PartitionedCall:output:0#^dropout_99/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_100_layer_call_and_return_conditional_losses_17156312%
#dropout_100/StatefulPartitionedCall
flatten_66/PartitionedCallPartitionedCall,dropout_100/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_66_layer_call_and_return_conditional_losses_17156552
flatten_66/PartitionedCallΊ
!dense_198/StatefulPartitionedCallStatefulPartitionedCall#flatten_66/PartitionedCall:output:0dense_198_1715685dense_198_1715687*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_198_layer_call_and_return_conditional_losses_17156742#
!dense_198/StatefulPartitionedCallΐ
!dense_199/StatefulPartitionedCallStatefulPartitionedCall*dense_198/StatefulPartitionedCall:output:0dense_199_1715712dense_199_1715714*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_199_layer_call_and_return_conditional_losses_17157012#
!dense_199/StatefulPartitionedCallΐ
!dense_200/StatefulPartitionedCallStatefulPartitionedCall*dense_199/StatefulPartitionedCall:output:0dense_200_1715739dense_200_1715741*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_200_layer_call_and_return_conditional_losses_17157282#
!dense_200/StatefulPartitionedCall³
IdentityIdentity*dense_200/StatefulPartitionedCall:output:03^Training_Data_Augmentation/StatefulPartitionedCall#^conv2d_100/StatefulPartitionedCall"^conv2d_99/StatefulPartitionedCall"^dense_198/StatefulPartitionedCall"^dense_199/StatefulPartitionedCall"^dense_200/StatefulPartitionedCall$^dropout_100/StatefulPartitionedCall#^dropout_99/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:?????????:::::::::::2h
2Training_Data_Augmentation/StatefulPartitionedCall2Training_Data_Augmentation/StatefulPartitionedCall2H
"conv2d_100/StatefulPartitionedCall"conv2d_100/StatefulPartitionedCall2F
!conv2d_99/StatefulPartitionedCall!conv2d_99/StatefulPartitionedCall2F
!dense_198/StatefulPartitionedCall!dense_198/StatefulPartitionedCall2F
!dense_199/StatefulPartitionedCall!dense_199/StatefulPartitionedCall2F
!dense_200/StatefulPartitionedCall!dense_200/StatefulPartitionedCall2J
#dropout_100/StatefulPartitionedCall#dropout_100/StatefulPartitionedCall2H
"dropout_99/StatefulPartitionedCall"dropout_99/StatefulPartitionedCall:q m
/
_output_shapes
:?????????
:
_user_specified_name" Training_Data_Augmentation_input
΄
?
F__inference_dense_198_layer_call_and_return_conditional_losses_1716502

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:?????????2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:::P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
γ

+__inference_dense_199_layer_call_fn_1716531

inputs
unknown
	unknown_0
identity’StatefulPartitionedCallφ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_199_layer_call_and_return_conditional_losses_17157012
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
φ
h
L__inference_random_contrast_layer_call_and_return_conditional_losses_1715440

inputs
identityb
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
(
g
H__inference_random_flip_layer_call_and_return_conditional_losses_1716579

inputs
identityΏ
)random_flip_left_right/control_dependencyIdentityinputs*
T0*
_class
loc:@inputs*/
_output_shapes
:?????????2+
)random_flip_left_right/control_dependency
random_flip_left_right/ShapeShape2random_flip_left_right/control_dependency:output:0*
T0*
_output_shapes
:2
random_flip_left_right/Shape’
*random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*random_flip_left_right/strided_slice/stack¦
,random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,random_flip_left_right/strided_slice/stack_1¦
,random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,random_flip_left_right/strided_slice/stack_2μ
$random_flip_left_right/strided_sliceStridedSlice%random_flip_left_right/Shape:output:03random_flip_left_right/strided_slice/stack:output:05random_flip_left_right/strided_slice/stack_1:output:05random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$random_flip_left_right/strided_sliceΏ
+random_flip_left_right/random_uniform/shapePack-random_flip_left_right/strided_slice:output:0*
N*
T0*
_output_shapes
:2-
+random_flip_left_right/random_uniform/shape
)random_flip_left_right/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2+
)random_flip_left_right/random_uniform/min
)random_flip_left_right/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2+
)random_flip_left_right/random_uniform/maxμ
3random_flip_left_right/random_uniform/RandomUniformRandomUniform4random_flip_left_right/random_uniform/shape:output:0*
T0*#
_output_shapes
:?????????*
dtype025
3random_flip_left_right/random_uniform/RandomUniformύ
)random_flip_left_right/random_uniform/MulMul<random_flip_left_right/random_uniform/RandomUniform:output:02random_flip_left_right/random_uniform/max:output:0*
T0*#
_output_shapes
:?????????2+
)random_flip_left_right/random_uniform/Mul
&random_flip_left_right/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2(
&random_flip_left_right/Reshape/shape/1
&random_flip_left_right/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2(
&random_flip_left_right/Reshape/shape/2
&random_flip_left_right/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2(
&random_flip_left_right/Reshape/shape/3Δ
$random_flip_left_right/Reshape/shapePack-random_flip_left_right/strided_slice:output:0/random_flip_left_right/Reshape/shape/1:output:0/random_flip_left_right/Reshape/shape/2:output:0/random_flip_left_right/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2&
$random_flip_left_right/Reshape/shapeγ
random_flip_left_right/ReshapeReshape-random_flip_left_right/random_uniform/Mul:z:0-random_flip_left_right/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2 
random_flip_left_right/Reshape¨
random_flip_left_right/RoundRound'random_flip_left_right/Reshape:output:0*
T0*/
_output_shapes
:?????????2
random_flip_left_right/Round
%random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:2'
%random_flip_left_right/ReverseV2/axisο
 random_flip_left_right/ReverseV2	ReverseV22random_flip_left_right/control_dependency:output:0.random_flip_left_right/ReverseV2/axis:output:0*
T0*/
_output_shapes
:?????????2"
 random_flip_left_right/ReverseV2Ζ
random_flip_left_right/mulMul random_flip_left_right/Round:y:0)random_flip_left_right/ReverseV2:output:0*
T0*/
_output_shapes
:?????????2
random_flip_left_right/mul
random_flip_left_right/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
random_flip_left_right/sub/xΒ
random_flip_left_right/subSub%random_flip_left_right/sub/x:output:0 random_flip_left_right/Round:y:0*
T0*/
_output_shapes
:?????????2
random_flip_left_right/subΡ
random_flip_left_right/mul_1Mulrandom_flip_left_right/sub:z:02random_flip_left_right/control_dependency:output:0*
T0*/
_output_shapes
:?????????2
random_flip_left_right/mul_1½
random_flip_left_right/addAddV2random_flip_left_right/mul:z:0 random_flip_left_right/mul_1:z:0*
T0*/
_output_shapes
:?????????2
random_flip_left_right/addz
IdentityIdentityrandom_flip_left_right/add:z:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
³0
«
U__inference_DoubleConv_LessDense_avg_layer_call_and_return_conditional_losses_1715780$
 training_data_augmentation_input
conv2d_99_1715749
conv2d_99_1715751
conv2d_100_1715756
conv2d_100_1715758
dense_198_1715764
dense_198_1715766
dense_199_1715769
dense_199_1715771
dense_200_1715774
dense_200_1715776
identity’"conv2d_100/StatefulPartitionedCall’!conv2d_99/StatefulPartitionedCall’!dense_198/StatefulPartitionedCall’!dense_199/StatefulPartitionedCall’!dense_200/StatefulPartitionedCall­
*Training_Data_Augmentation/PartitionedCallPartitionedCall training_data_augmentation_input*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *`
f[RY
W__inference_Training_Data_Augmentation_layer_call_and_return_conditional_losses_17154872,
*Training_Data_Augmentation/PartitionedCallΡ
!conv2d_99/StatefulPartitionedCallStatefulPartitionedCall3Training_Data_Augmentation/PartitionedCall:output:0conv2d_99_1715749conv2d_99_1715751*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_conv2d_99_layer_call_and_return_conditional_losses_17155442#
!conv2d_99/StatefulPartitionedCall₯
$average_pooling2d_44/PartitionedCallPartitionedCall*conv2d_99/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_average_pooling2d_44_layer_call_and_return_conditional_losses_17154962&
$average_pooling2d_44/PartitionedCall
dropout_99/PartitionedCallPartitionedCall-average_pooling2d_44/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_99_layer_call_and_return_conditional_losses_17155782
dropout_99/PartitionedCallΖ
"conv2d_100/StatefulPartitionedCallStatefulPartitionedCall#dropout_99/PartitionedCall:output:0conv2d_100_1715756conv2d_100_1715758*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_conv2d_100_layer_call_and_return_conditional_losses_17156022$
"conv2d_100/StatefulPartitionedCall¦
$average_pooling2d_45/PartitionedCallPartitionedCall+conv2d_100/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_average_pooling2d_45_layer_call_and_return_conditional_losses_17155082&
$average_pooling2d_45/PartitionedCall
dropout_100/PartitionedCallPartitionedCall-average_pooling2d_45/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_100_layer_call_and_return_conditional_losses_17156362
dropout_100/PartitionedCallϊ
flatten_66/PartitionedCallPartitionedCall$dropout_100/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_66_layer_call_and_return_conditional_losses_17156552
flatten_66/PartitionedCallΊ
!dense_198/StatefulPartitionedCallStatefulPartitionedCall#flatten_66/PartitionedCall:output:0dense_198_1715764dense_198_1715766*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_198_layer_call_and_return_conditional_losses_17156742#
!dense_198/StatefulPartitionedCallΐ
!dense_199/StatefulPartitionedCallStatefulPartitionedCall*dense_198/StatefulPartitionedCall:output:0dense_199_1715769dense_199_1715771*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_199_layer_call_and_return_conditional_losses_17157012#
!dense_199/StatefulPartitionedCallΐ
!dense_200/StatefulPartitionedCallStatefulPartitionedCall*dense_199/StatefulPartitionedCall:output:0dense_200_1715774dense_200_1715776*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_200_layer_call_and_return_conditional_losses_17157282#
!dense_200/StatefulPartitionedCall³
IdentityIdentity*dense_200/StatefulPartitionedCall:output:0#^conv2d_100/StatefulPartitionedCall"^conv2d_99/StatefulPartitionedCall"^dense_198/StatefulPartitionedCall"^dense_199/StatefulPartitionedCall"^dense_200/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????::::::::::2H
"conv2d_100/StatefulPartitionedCall"conv2d_100/StatefulPartitionedCall2F
!conv2d_99/StatefulPartitionedCall!conv2d_99/StatefulPartitionedCall2F
!dense_198/StatefulPartitionedCall!dense_198/StatefulPartitionedCall2F
!dense_199/StatefulPartitionedCall!dense_199/StatefulPartitionedCall2F
!dense_200/StatefulPartitionedCall!dense_200/StatefulPartitionedCall:q m
/
_output_shapes
:?????????
:
_user_specified_name" Training_Data_Augmentation_input
ς
d
H__inference_random_flip_layer_call_and_return_conditional_losses_1715284

inputs
identityb
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
(
g
H__inference_random_flip_layer_call_and_return_conditional_losses_1715280

inputs
identityΏ
)random_flip_left_right/control_dependencyIdentityinputs*
T0*
_class
loc:@inputs*/
_output_shapes
:?????????2+
)random_flip_left_right/control_dependency
random_flip_left_right/ShapeShape2random_flip_left_right/control_dependency:output:0*
T0*
_output_shapes
:2
random_flip_left_right/Shape’
*random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*random_flip_left_right/strided_slice/stack¦
,random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,random_flip_left_right/strided_slice/stack_1¦
,random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,random_flip_left_right/strided_slice/stack_2μ
$random_flip_left_right/strided_sliceStridedSlice%random_flip_left_right/Shape:output:03random_flip_left_right/strided_slice/stack:output:05random_flip_left_right/strided_slice/stack_1:output:05random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$random_flip_left_right/strided_sliceΏ
+random_flip_left_right/random_uniform/shapePack-random_flip_left_right/strided_slice:output:0*
N*
T0*
_output_shapes
:2-
+random_flip_left_right/random_uniform/shape
)random_flip_left_right/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2+
)random_flip_left_right/random_uniform/min
)random_flip_left_right/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2+
)random_flip_left_right/random_uniform/maxμ
3random_flip_left_right/random_uniform/RandomUniformRandomUniform4random_flip_left_right/random_uniform/shape:output:0*
T0*#
_output_shapes
:?????????*
dtype025
3random_flip_left_right/random_uniform/RandomUniformύ
)random_flip_left_right/random_uniform/MulMul<random_flip_left_right/random_uniform/RandomUniform:output:02random_flip_left_right/random_uniform/max:output:0*
T0*#
_output_shapes
:?????????2+
)random_flip_left_right/random_uniform/Mul
&random_flip_left_right/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2(
&random_flip_left_right/Reshape/shape/1
&random_flip_left_right/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2(
&random_flip_left_right/Reshape/shape/2
&random_flip_left_right/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2(
&random_flip_left_right/Reshape/shape/3Δ
$random_flip_left_right/Reshape/shapePack-random_flip_left_right/strided_slice:output:0/random_flip_left_right/Reshape/shape/1:output:0/random_flip_left_right/Reshape/shape/2:output:0/random_flip_left_right/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2&
$random_flip_left_right/Reshape/shapeγ
random_flip_left_right/ReshapeReshape-random_flip_left_right/random_uniform/Mul:z:0-random_flip_left_right/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2 
random_flip_left_right/Reshape¨
random_flip_left_right/RoundRound'random_flip_left_right/Reshape:output:0*
T0*/
_output_shapes
:?????????2
random_flip_left_right/Round
%random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:2'
%random_flip_left_right/ReverseV2/axisο
 random_flip_left_right/ReverseV2	ReverseV22random_flip_left_right/control_dependency:output:0.random_flip_left_right/ReverseV2/axis:output:0*
T0*/
_output_shapes
:?????????2"
 random_flip_left_right/ReverseV2Ζ
random_flip_left_right/mulMul random_flip_left_right/Round:y:0)random_flip_left_right/ReverseV2:output:0*
T0*/
_output_shapes
:?????????2
random_flip_left_right/mul
random_flip_left_right/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
random_flip_left_right/sub/xΒ
random_flip_left_right/subSub%random_flip_left_right/sub/x:output:0 random_flip_left_right/Round:y:0*
T0*/
_output_shapes
:?????????2
random_flip_left_right/subΡ
random_flip_left_right/mul_1Mulrandom_flip_left_right/sub:z:02random_flip_left_right/control_dependency:output:0*
T0*/
_output_shapes
:?????????2
random_flip_left_right/mul_1½
random_flip_left_right/addAddV2random_flip_left_right/mul:z:0 random_flip_left_right/mul_1:z:0*
T0*/
_output_shapes
:?????????2
random_flip_left_right/addz
IdentityIdentityrandom_flip_left_right/add:z:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
έ	
‘
:__inference_DoubleConv_LessDense_avg_layer_call_fn_1715905$
 training_data_augmentation_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCall training_data_augmentation_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *^
fYRW
U__inference_DoubleConv_LessDense_avg_layer_call_and_return_conditional_losses_17158822
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:q m
/
_output_shapes
:?????????
:
_user_specified_name" Training_Data_Augmentation_input
Ε
f
G__inference_dropout_99_layer_call_and_return_conditional_losses_1716418

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nΫΆ?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeΌ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2
dropout/GreaterEqual/yΖ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????@2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????@2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
Ζ
f
-__inference_dropout_100_layer_call_fn_1716475

inputs
identity’StatefulPartitionedCallζ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_100_layer_call_and_return_conditional_losses_17156312
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
Ί
I
-__inference_random_flip_layer_call_fn_1716593

inputs
identityΞ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_random_flip_layer_call_and_return_conditional_losses_17152842
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs"ΈL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ζ
serving_default?
u
 Training_Data_Augmentation_inputQ
2serving_default_Training_Data_Augmentation_input:0?????????=
	dense_2000
StatefulPartitionedCall:0?????????
tensorflow/serving/predict:·
ΣV
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer_with_weights-1
layer-4
layer-5
layer-6
layer-7
	layer_with_weights-2
	layer-8

layer_with_weights-3

layer-9
layer_with_weights-4
layer-10
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
Κ__call__
Λ_default_save_signature
+Μ&call_and_return_all_conditional_losses"σR
_tf_keras_sequentialΤR{"class_name": "Sequential", "name": "DoubleConv_LessDense_avg", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "DoubleConv_LessDense_avg", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Training_Data_Augmentation_input"}}, {"class_name": "Sequential", "config": {"name": "Training_Data_Augmentation", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "random_flip_input"}}, {"class_name": "RandomFlip", "config": {"name": "random_flip", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "mode": "horizontal", "seed": null}}, {"class_name": "RandomRotation", "config": {"name": "random_rotation", "trainable": true, "dtype": "float32", "factor": 0.2, "fill_mode": "reflect", "interpolation": "bilinear", "seed": null}}, {"class_name": "RandomContrast", "config": {"name": "random_contrast", "trainable": true, "dtype": "float32", "factor": 0.1, "seed": null}}]}}, {"class_name": "Conv2D", "config": {"name": "conv2d_99", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "AveragePooling2D", "config": {"name": "average_pooling2d_44", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_99", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_100", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "AveragePooling2D", "config": {"name": "average_pooling2d_45", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_100", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten_66", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_198", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_199", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_200", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "DoubleConv_LessDense_avg", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Training_Data_Augmentation_input"}}, {"class_name": "Sequential", "config": {"name": "Training_Data_Augmentation", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "random_flip_input"}}, {"class_name": "RandomFlip", "config": {"name": "random_flip", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "mode": "horizontal", "seed": null}}, {"class_name": "RandomRotation", "config": {"name": "random_rotation", "trainable": true, "dtype": "float32", "factor": 0.2, "fill_mode": "reflect", "interpolation": "bilinear", "seed": null}}, {"class_name": "RandomContrast", "config": {"name": "random_contrast", "trainable": true, "dtype": "float32", "factor": 0.1, "seed": null}}]}}, {"class_name": "Conv2D", "config": {"name": "conv2d_99", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "AveragePooling2D", "config": {"name": "average_pooling2d_44", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_99", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_100", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "AveragePooling2D", "config": {"name": "average_pooling2d_45", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_100", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten_66", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_198", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_199", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_200", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "sparse_categorical_crossentropy", "metrics": ["accuracy"], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.009999999776482582, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
Ϋ
layer-0
layer-1
layer-2
regularization_losses
	variables
trainable_variables
	keras_api
Ν__call__
+Ξ&call_and_return_all_conditional_losses"£
_tf_keras_sequential{"class_name": "Sequential", "name": "Training_Data_Augmentation", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "Training_Data_Augmentation", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "random_flip_input"}}, {"class_name": "RandomFlip", "config": {"name": "random_flip", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "mode": "horizontal", "seed": null}}, {"class_name": "RandomRotation", "config": {"name": "random_rotation", "trainable": true, "dtype": "float32", "factor": 0.2, "fill_mode": "reflect", "interpolation": "bilinear", "seed": null}}, {"class_name": "RandomContrast", "config": {"name": "random_contrast", "trainable": true, "dtype": "float32", "factor": 0.1, "seed": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "Training_Data_Augmentation", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "random_flip_input"}}, {"class_name": "RandomFlip", "config": {"name": "random_flip", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "mode": "horizontal", "seed": null}}, {"class_name": "RandomRotation", "config": {"name": "random_rotation", "trainable": true, "dtype": "float32", "factor": 0.2, "fill_mode": "reflect", "interpolation": "bilinear", "seed": null}}, {"class_name": "RandomContrast", "config": {"name": "random_contrast", "trainable": true, "dtype": "float32", "factor": 0.1, "seed": null}}]}}}
φ


kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
Ο__call__
+Π&call_and_return_all_conditional_losses"Ο	
_tf_keras_layer΅	{"class_name": "Conv2D", "name": "conv2d_99", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_99", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 1]}}

regularization_losses
 	variables
!trainable_variables
"	keras_api
Ρ__call__
+?&call_and_return_all_conditional_losses"ώ
_tf_keras_layerδ{"class_name": "AveragePooling2D", "name": "average_pooling2d_44", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "average_pooling2d_44", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ι
#regularization_losses
$	variables
%trainable_variables
&	keras_api
Σ__call__
+Τ&call_and_return_all_conditional_losses"Ψ
_tf_keras_layerΎ{"class_name": "Dropout", "name": "dropout_99", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_99", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}}
ω	

'kernel
(bias
)regularization_losses
*	variables
+trainable_variables
,	keras_api
Υ__call__
+Φ&call_and_return_all_conditional_losses"?
_tf_keras_layerΈ{"class_name": "Conv2D", "name": "conv2d_100", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_100", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 13, 13, 64]}}

-regularization_losses
.	variables
/trainable_variables
0	keras_api
Χ__call__
+Ψ&call_and_return_all_conditional_losses"ώ
_tf_keras_layerδ{"class_name": "AveragePooling2D", "name": "average_pooling2d_45", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "average_pooling2d_45", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
λ
1regularization_losses
2	variables
3trainable_variables
4	keras_api
Ω__call__
+Ϊ&call_and_return_all_conditional_losses"Ϊ
_tf_keras_layerΐ{"class_name": "Dropout", "name": "dropout_100", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_100", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}}
κ
5regularization_losses
6	variables
7trainable_variables
8	keras_api
Ϋ__call__
+ά&call_and_return_all_conditional_losses"Ω
_tf_keras_layerΏ{"class_name": "Flatten", "name": "flatten_66", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_66", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
ϋ

9kernel
:bias
;regularization_losses
<	variables
=trainable_variables
>	keras_api
έ__call__
+ή&call_and_return_all_conditional_losses"Τ
_tf_keras_layerΊ{"class_name": "Dense", "name": "dense_198", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_198", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2304}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2304]}}
ψ

?kernel
@bias
Aregularization_losses
B	variables
Ctrainable_variables
D	keras_api
ί__call__
+ΰ&call_and_return_all_conditional_losses"Ρ
_tf_keras_layer·{"class_name": "Dense", "name": "dense_199", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_199", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
ω

Ekernel
Fbias
Gregularization_losses
H	variables
Itrainable_variables
J	keras_api
α__call__
+β&call_and_return_all_conditional_losses"?
_tf_keras_layerΈ{"class_name": "Dense", "name": "dense_200", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_200", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}

Kiter

Lbeta_1

Mbeta_2
	Ndecay
Olearning_ratemΆm·'mΈ(mΉ9mΊ:m»?mΌ@m½EmΎFmΏvΐvΑ'vΒ(vΓ9vΔ:vΕ?vΖ@vΗEvΘFvΙ"
	optimizer
 "
trackable_list_wrapper
f
0
1
'2
(3
94
:5
?6
@7
E8
F9"
trackable_list_wrapper
f
0
1
'2
(3
94
:5
?6
@7
E8
F9"
trackable_list_wrapper
Ξ

Players
Qlayer_metrics
Rnon_trainable_variables
regularization_losses
	variables
trainable_variables
Smetrics
Tlayer_regularization_losses
Κ__call__
Λ_default_save_signature
+Μ&call_and_return_all_conditional_losses
'Μ"call_and_return_conditional_losses"
_generic_user_object
-
γserving_default"
signature_map
ύ
U_rng
Vregularization_losses
W	variables
Xtrainable_variables
Y	keras_api
δ__call__
+ε&call_and_return_all_conditional_losses"β
_tf_keras_layerΘ{"class_name": "RandomFlip", "name": "random_flip", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "random_flip", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "mode": "horizontal", "seed": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
¦
Z_rng
[regularization_losses
\	variables
]trainable_variables
^	keras_api
ζ__call__
+η&call_and_return_all_conditional_losses"
_tf_keras_layerρ{"class_name": "RandomRotation", "name": "random_rotation", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "random_rotation", "trainable": true, "dtype": "float32", "factor": 0.2, "fill_mode": "reflect", "interpolation": "bilinear", "seed": null}}
η
_regularization_losses
`	variables
atrainable_variables
b	keras_api
θ__call__
+ι&call_and_return_all_conditional_losses"Φ
_tf_keras_layerΌ{"class_name": "RandomContrast", "name": "random_contrast", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "random_contrast", "trainable": true, "dtype": "float32", "factor": 0.1, "seed": null}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°

clayers
dlayer_metrics
enon_trainable_variables
regularization_losses
	variables
trainable_variables
fmetrics
glayer_regularization_losses
Ν__call__
+Ξ&call_and_return_all_conditional_losses
'Ξ"call_and_return_conditional_losses"
_generic_user_object
*:(@2conv2d_99/kernel
:@2conv2d_99/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
°

hlayers
ilayer_metrics
jnon_trainable_variables
regularization_losses
	variables
trainable_variables
kmetrics
llayer_regularization_losses
Ο__call__
+Π&call_and_return_all_conditional_losses
'Π"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°

mlayers
nlayer_metrics
onon_trainable_variables
regularization_losses
 	variables
!trainable_variables
pmetrics
qlayer_regularization_losses
Ρ__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°

rlayers
slayer_metrics
tnon_trainable_variables
#regularization_losses
$	variables
%trainable_variables
umetrics
vlayer_regularization_losses
Σ__call__
+Τ&call_and_return_all_conditional_losses
'Τ"call_and_return_conditional_losses"
_generic_user_object
+:)@@2conv2d_100/kernel
:@2conv2d_100/bias
 "
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
°

wlayers
xlayer_metrics
ynon_trainable_variables
)regularization_losses
*	variables
+trainable_variables
zmetrics
{layer_regularization_losses
Υ__call__
+Φ&call_and_return_all_conditional_losses
'Φ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
±

|layers
}layer_metrics
~non_trainable_variables
-regularization_losses
.	variables
/trainable_variables
metrics
 layer_regularization_losses
Χ__call__
+Ψ&call_and_return_all_conditional_losses
'Ψ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
΅
layers
layer_metrics
non_trainable_variables
1regularization_losses
2	variables
3trainable_variables
metrics
 layer_regularization_losses
Ω__call__
+Ϊ&call_and_return_all_conditional_losses
'Ϊ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
΅
layers
layer_metrics
non_trainable_variables
5regularization_losses
6	variables
7trainable_variables
metrics
 layer_regularization_losses
Ϋ__call__
+ά&call_and_return_all_conditional_losses
'ά"call_and_return_conditional_losses"
_generic_user_object
$:"
2dense_198/kernel
:2dense_198/bias
 "
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
΅
layers
layer_metrics
non_trainable_variables
;regularization_losses
<	variables
=trainable_variables
metrics
 layer_regularization_losses
έ__call__
+ή&call_and_return_all_conditional_losses
'ή"call_and_return_conditional_losses"
_generic_user_object
#:!	@2dense_199/kernel
:@2dense_199/bias
 "
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
΅
layers
layer_metrics
non_trainable_variables
Aregularization_losses
B	variables
Ctrainable_variables
metrics
 layer_regularization_losses
ί__call__
+ΰ&call_and_return_all_conditional_losses
'ΰ"call_and_return_conditional_losses"
_generic_user_object
": @
2dense_200/kernel
:
2dense_200/bias
 "
trackable_list_wrapper
.
E0
F1"
trackable_list_wrapper
.
E0
F1"
trackable_list_wrapper
΅
layers
layer_metrics
non_trainable_variables
Gregularization_losses
H	variables
Itrainable_variables
metrics
 layer_regularization_losses
α__call__
+β&call_and_return_all_conditional_losses
'β"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
n
0
1
2
3
4
5
6
7
	8

9
10"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
/

_state_var"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
΅
layers
layer_metrics
non_trainable_variables
Vregularization_losses
W	variables
Xtrainable_variables
 metrics
 ‘layer_regularization_losses
δ__call__
+ε&call_and_return_all_conditional_losses
'ε"call_and_return_conditional_losses"
_generic_user_object
/
’
_state_var"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
΅
£layers
€layer_metrics
₯non_trainable_variables
[regularization_losses
\	variables
]trainable_variables
¦metrics
 §layer_regularization_losses
ζ__call__
+η&call_and_return_all_conditional_losses
'η"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
΅
¨layers
©layer_metrics
ͺnon_trainable_variables
_regularization_losses
`	variables
atrainable_variables
«metrics
 ¬layer_regularization_losses
θ__call__
+ι&call_and_return_all_conditional_losses
'ι"call_and_return_conditional_losses"
_generic_user_object
5
0
1
2"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Ώ

­total

?count
―	variables
°	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}


±total

²count
³
_fn_kwargs
΄	variables
΅	keras_api"Ώ
_tf_keras_metric€{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}}
:	2Variable
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:	2Variable
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
0
­0
?1"
trackable_list_wrapper
.
―	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
±0
²1"
trackable_list_wrapper
.
΄	variables"
_generic_user_object
/:-@2Adam/conv2d_99/kernel/m
!:@2Adam/conv2d_99/bias/m
0:.@@2Adam/conv2d_100/kernel/m
": @2Adam/conv2d_100/bias/m
):'
2Adam/dense_198/kernel/m
": 2Adam/dense_198/bias/m
(:&	@2Adam/dense_199/kernel/m
!:@2Adam/dense_199/bias/m
':%@
2Adam/dense_200/kernel/m
!:
2Adam/dense_200/bias/m
/:-@2Adam/conv2d_99/kernel/v
!:@2Adam/conv2d_99/bias/v
0:.@@2Adam/conv2d_100/kernel/v
": @2Adam/conv2d_100/bias/v
):'
2Adam/dense_198/kernel/v
": 2Adam/dense_198/bias/v
(:&	@2Adam/dense_199/kernel/v
!:@2Adam/dense_199/bias/v
':%@
2Adam/dense_200/kernel/v
!:
2Adam/dense_200/bias/v
Ά2³
:__inference_DoubleConv_LessDense_avg_layer_call_fn_1716206
:__inference_DoubleConv_LessDense_avg_layer_call_fn_1715905
:__inference_DoubleConv_LessDense_avg_layer_call_fn_1716231
:__inference_DoubleConv_LessDense_avg_layer_call_fn_1715845ΐ
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
2ώ
"__inference__wrapped_model_1715125Χ
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *G’D
B?
 Training_Data_Augmentation_input?????????
’2
U__inference_DoubleConv_LessDense_avg_layer_call_and_return_conditional_losses_1716179
U__inference_DoubleConv_LessDense_avg_layer_call_and_return_conditional_losses_1716134
U__inference_DoubleConv_LessDense_avg_layer_call_and_return_conditional_losses_1715745
U__inference_DoubleConv_LessDense_avg_layer_call_and_return_conditional_losses_1715780ΐ
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
Ύ2»
<__inference_Training_Data_Augmentation_layer_call_fn_1715478
<__inference_Training_Data_Augmentation_layer_call_fn_1716386
<__inference_Training_Data_Augmentation_layer_call_fn_1716381
<__inference_Training_Data_Augmentation_layer_call_fn_1715490ΐ
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
ͺ2§
W__inference_Training_Data_Augmentation_layer_call_and_return_conditional_losses_1715454
W__inference_Training_Data_Augmentation_layer_call_and_return_conditional_losses_1715461
W__inference_Training_Data_Augmentation_layer_call_and_return_conditional_losses_1716374
W__inference_Training_Data_Augmentation_layer_call_and_return_conditional_losses_1716370ΐ
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
Υ2?
+__inference_conv2d_99_layer_call_fn_1716406’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
π2ν
F__inference_conv2d_99_layer_call_and_return_conditional_losses_1716397’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
2
6__inference_average_pooling2d_44_layer_call_fn_1715502ΰ
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *@’=
;84????????????????????????????????????
Ή2Ά
Q__inference_average_pooling2d_44_layer_call_and_return_conditional_losses_1715496ΰ
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *@’=
;84????????????????????????????????????
2
,__inference_dropout_99_layer_call_fn_1716433
,__inference_dropout_99_layer_call_fn_1716428΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
Μ2Ι
G__inference_dropout_99_layer_call_and_return_conditional_losses_1716418
G__inference_dropout_99_layer_call_and_return_conditional_losses_1716423΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
Φ2Σ
,__inference_conv2d_100_layer_call_fn_1716453’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ρ2ξ
G__inference_conv2d_100_layer_call_and_return_conditional_losses_1716444’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
2
6__inference_average_pooling2d_45_layer_call_fn_1715514ΰ
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *@’=
;84????????????????????????????????????
Ή2Ά
Q__inference_average_pooling2d_45_layer_call_and_return_conditional_losses_1715508ΰ
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *@’=
;84????????????????????????????????????
2
-__inference_dropout_100_layer_call_fn_1716480
-__inference_dropout_100_layer_call_fn_1716475΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
Ξ2Λ
H__inference_dropout_100_layer_call_and_return_conditional_losses_1716470
H__inference_dropout_100_layer_call_and_return_conditional_losses_1716465΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
Φ2Σ
,__inference_flatten_66_layer_call_fn_1716491’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ρ2ξ
G__inference_flatten_66_layer_call_and_return_conditional_losses_1716486’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Υ2?
+__inference_dense_198_layer_call_fn_1716511’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
π2ν
F__inference_dense_198_layer_call_and_return_conditional_losses_1716502’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Υ2?
+__inference_dense_199_layer_call_fn_1716531’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
π2ν
F__inference_dense_199_layer_call_and_return_conditional_losses_1716522’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Υ2?
+__inference_dense_200_layer_call_fn_1716551’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
π2ν
F__inference_dense_200_layer_call_and_return_conditional_losses_1716542’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
MBK
%__inference_signature_wrapper_1715940 Training_Data_Augmentation_input
φ2σ
-__inference_random_flip_layer_call_fn_1716588
-__inference_random_flip_layer_call_fn_1716659
-__inference_random_flip_layer_call_fn_1716654
-__inference_random_flip_layer_call_fn_1716593΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
β2ί
H__inference_random_flip_layer_call_and_return_conditional_losses_1716649
H__inference_random_flip_layer_call_and_return_conditional_losses_1716579
H__inference_random_flip_layer_call_and_return_conditional_losses_1716583
H__inference_random_flip_layer_call_and_return_conditional_losses_1716645΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
 2
1__inference_random_rotation_layer_call_fn_1716776
1__inference_random_rotation_layer_call_fn_1716781΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
Φ2Σ
L__inference_random_rotation_layer_call_and_return_conditional_losses_1716769
L__inference_random_rotation_layer_call_and_return_conditional_losses_1716765΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
 2
1__inference_random_contrast_layer_call_fn_1716803
1__inference_random_contrast_layer_call_fn_1716808΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
Φ2Σ
L__inference_random_contrast_layer_call_and_return_conditional_losses_1716798
L__inference_random_contrast_layer_call_and_return_conditional_losses_1716794΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 κ
U__inference_DoubleConv_LessDense_avg_layer_call_and_return_conditional_losses_1715745’'(9:?@EFY’V
O’L
B?
 Training_Data_Augmentation_input?????????
p

 
ͺ "%’"

0?????????

 θ
U__inference_DoubleConv_LessDense_avg_layer_call_and_return_conditional_losses_1715780
'(9:?@EFY’V
O’L
B?
 Training_Data_Augmentation_input?????????
p 

 
ͺ "%’"

0?????????

 Ο
U__inference_DoubleConv_LessDense_avg_layer_call_and_return_conditional_losses_1716134v’'(9:?@EF?’<
5’2
(%
inputs?????????
p

 
ͺ "%’"

0?????????

 Ν
U__inference_DoubleConv_LessDense_avg_layer_call_and_return_conditional_losses_1716179t
'(9:?@EF?’<
5’2
(%
inputs?????????
p 

 
ͺ "%’"

0?????????

 Β
:__inference_DoubleConv_LessDense_avg_layer_call_fn_1715845’'(9:?@EFY’V
O’L
B?
 Training_Data_Augmentation_input?????????
p

 
ͺ "?????????
ΐ
:__inference_DoubleConv_LessDense_avg_layer_call_fn_1715905
'(9:?@EFY’V
O’L
B?
 Training_Data_Augmentation_input?????????
p 

 
ͺ "?????????
§
:__inference_DoubleConv_LessDense_avg_layer_call_fn_1716206i’'(9:?@EF?’<
5’2
(%
inputs?????????
p

 
ͺ "?????????
₯
:__inference_DoubleConv_LessDense_avg_layer_call_fn_1716231g
'(9:?@EF?’<
5’2
(%
inputs?????????
p 

 
ͺ "?????????
Ϊ
W__inference_Training_Data_Augmentation_layer_call_and_return_conditional_losses_1715454’J’G
@’=
30
random_flip_input?????????
p

 
ͺ "-’*
# 
0?????????
 Φ
W__inference_Training_Data_Augmentation_layer_call_and_return_conditional_losses_1715461{J’G
@’=
30
random_flip_input?????????
p 

 
ͺ "-’*
# 
0?????????
 Ο
W__inference_Training_Data_Augmentation_layer_call_and_return_conditional_losses_1716370t’?’<
5’2
(%
inputs?????????
p

 
ͺ "-’*
# 
0?????????
 Λ
W__inference_Training_Data_Augmentation_layer_call_and_return_conditional_losses_1716374p?’<
5’2
(%
inputs?????????
p 

 
ͺ "-’*
# 
0?????????
 ²
<__inference_Training_Data_Augmentation_layer_call_fn_1715478r’J’G
@’=
30
random_flip_input?????????
p

 
ͺ " ??????????
<__inference_Training_Data_Augmentation_layer_call_fn_1715490nJ’G
@’=
30
random_flip_input?????????
p 

 
ͺ " ?????????§
<__inference_Training_Data_Augmentation_layer_call_fn_1716381g’?’<
5’2
(%
inputs?????????
p

 
ͺ " ?????????£
<__inference_Training_Data_Augmentation_layer_call_fn_1716386c?’<
5’2
(%
inputs?????????
p 

 
ͺ " ?????????½
"__inference__wrapped_model_1715125
'(9:?@EFQ’N
G’D
B?
 Training_Data_Augmentation_input?????????
ͺ "5ͺ2
0
	dense_200# 
	dense_200?????????
τ
Q__inference_average_pooling2d_44_layer_call_and_return_conditional_losses_1715496R’O
H’E
C@
inputs4????????????????????????????????????
ͺ "H’E
>;
04????????????????????????????????????
 Μ
6__inference_average_pooling2d_44_layer_call_fn_1715502R’O
H’E
C@
inputs4????????????????????????????????????
ͺ ";84????????????????????????????????????τ
Q__inference_average_pooling2d_45_layer_call_and_return_conditional_losses_1715508R’O
H’E
C@
inputs4????????????????????????????????????
ͺ "H’E
>;
04????????????????????????????????????
 Μ
6__inference_average_pooling2d_45_layer_call_fn_1715514R’O
H’E
C@
inputs4????????????????????????????????????
ͺ ";84????????????????????????????????????·
G__inference_conv2d_100_layer_call_and_return_conditional_losses_1716444l'(7’4
-’*
(%
inputs?????????@
ͺ "-’*
# 
0?????????@
 
,__inference_conv2d_100_layer_call_fn_1716453_'(7’4
-’*
(%
inputs?????????@
ͺ " ?????????@Ά
F__inference_conv2d_99_layer_call_and_return_conditional_losses_1716397l7’4
-’*
(%
inputs?????????
ͺ "-’*
# 
0?????????@
 
+__inference_conv2d_99_layer_call_fn_1716406_7’4
-’*
(%
inputs?????????
ͺ " ?????????@¨
F__inference_dense_198_layer_call_and_return_conditional_losses_1716502^9:0’-
&’#
!
inputs?????????
ͺ "&’#

0?????????
 
+__inference_dense_198_layer_call_fn_1716511Q9:0’-
&’#
!
inputs?????????
ͺ "?????????§
F__inference_dense_199_layer_call_and_return_conditional_losses_1716522]?@0’-
&’#
!
inputs?????????
ͺ "%’"

0?????????@
 
+__inference_dense_199_layer_call_fn_1716531P?@0’-
&’#
!
inputs?????????
ͺ "?????????@¦
F__inference_dense_200_layer_call_and_return_conditional_losses_1716542\EF/’,
%’"
 
inputs?????????@
ͺ "%’"

0?????????

 ~
+__inference_dense_200_layer_call_fn_1716551OEF/’,
%’"
 
inputs?????????@
ͺ "?????????
Έ
H__inference_dropout_100_layer_call_and_return_conditional_losses_1716465l;’8
1’.
(%
inputs?????????@
p
ͺ "-’*
# 
0?????????@
 Έ
H__inference_dropout_100_layer_call_and_return_conditional_losses_1716470l;’8
1’.
(%
inputs?????????@
p 
ͺ "-’*
# 
0?????????@
 
-__inference_dropout_100_layer_call_fn_1716475_;’8
1’.
(%
inputs?????????@
p
ͺ " ?????????@
-__inference_dropout_100_layer_call_fn_1716480_;’8
1’.
(%
inputs?????????@
p 
ͺ " ?????????@·
G__inference_dropout_99_layer_call_and_return_conditional_losses_1716418l;’8
1’.
(%
inputs?????????@
p
ͺ "-’*
# 
0?????????@
 ·
G__inference_dropout_99_layer_call_and_return_conditional_losses_1716423l;’8
1’.
(%
inputs?????????@
p 
ͺ "-’*
# 
0?????????@
 
,__inference_dropout_99_layer_call_fn_1716428_;’8
1’.
(%
inputs?????????@
p
ͺ " ?????????@
,__inference_dropout_99_layer_call_fn_1716433_;’8
1’.
(%
inputs?????????@
p 
ͺ " ?????????@¬
G__inference_flatten_66_layer_call_and_return_conditional_losses_1716486a7’4
-’*
(%
inputs?????????@
ͺ "&’#

0?????????
 
,__inference_flatten_66_layer_call_fn_1716491T7’4
-’*
(%
inputs?????????@
ͺ "?????????Ό
L__inference_random_contrast_layer_call_and_return_conditional_losses_1716794l;’8
1’.
(%
inputs?????????
p
ͺ "-’*
# 
0?????????
 Ό
L__inference_random_contrast_layer_call_and_return_conditional_losses_1716798l;’8
1’.
(%
inputs?????????
p 
ͺ "-’*
# 
0?????????
 
1__inference_random_contrast_layer_call_fn_1716803_;’8
1’.
(%
inputs?????????
p
ͺ " ?????????
1__inference_random_contrast_layer_call_fn_1716808_;’8
1’.
(%
inputs?????????
p 
ͺ " ?????????Έ
H__inference_random_flip_layer_call_and_return_conditional_losses_1716579l;’8
1’.
(%
inputs?????????
p
ͺ "-’*
# 
0?????????
 Έ
H__inference_random_flip_layer_call_and_return_conditional_losses_1716583l;’8
1’.
(%
inputs?????????
p 
ͺ "-’*
# 
0?????????
 ο
H__inference_random_flip_layer_call_and_return_conditional_losses_1716645’V’S
L’I
C@
inputs4????????????????????????????????????
p
ͺ "H’E
>;
04????????????????????????????????????
 ο
H__inference_random_flip_layer_call_and_return_conditional_losses_1716649’V’S
L’I
C@
inputs4????????????????????????????????????
p 
ͺ "H’E
>;
04????????????????????????????????????
 
-__inference_random_flip_layer_call_fn_1716588_;’8
1’.
(%
inputs?????????
p
ͺ " ?????????
-__inference_random_flip_layer_call_fn_1716593_;’8
1’.
(%
inputs?????????
p 
ͺ " ?????????Η
-__inference_random_flip_layer_call_fn_1716654V’S
L’I
C@
inputs4????????????????????????????????????
p
ͺ ";84????????????????????????????????????Η
-__inference_random_flip_layer_call_fn_1716659V’S
L’I
C@
inputs4????????????????????????????????????
p 
ͺ ";84????????????????????????????????????ΐ
L__inference_random_rotation_layer_call_and_return_conditional_losses_1716765p’;’8
1’.
(%
inputs?????????
p
ͺ "-’*
# 
0?????????
 Ό
L__inference_random_rotation_layer_call_and_return_conditional_losses_1716769l;’8
1’.
(%
inputs?????????
p 
ͺ "-’*
# 
0?????????
 
1__inference_random_rotation_layer_call_fn_1716776c’;’8
1’.
(%
inputs?????????
p
ͺ " ?????????
1__inference_random_rotation_layer_call_fn_1716781_;’8
1’.
(%
inputs?????????
p 
ͺ " ?????????δ
%__inference_signature_wrapper_1715940Ί
'(9:?@EFu’r
’ 
kͺh
f
 Training_Data_Augmentation_inputB?
 Training_Data_Augmentation_input?????????"5ͺ2
0
	dense_200# 
	dense_200?????????
