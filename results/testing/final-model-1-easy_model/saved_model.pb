??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
?
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
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.12v2.4.0-49-g85c8b2a817f8??
?
conv2d_77/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_77/kernel
}
$conv2d_77/kernel/Read/ReadVariableOpReadVariableOpconv2d_77/kernel*&
_output_shapes
: *
dtype0
t
conv2d_77/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_77/bias
m
"conv2d_77/bias/Read/ReadVariableOpReadVariableOpconv2d_77/bias*
_output_shapes
: *
dtype0
~
dense_165/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?*?*!
shared_namedense_165/kernel
w
$dense_165/kernel/Read/ReadVariableOpReadVariableOpdense_165/kernel* 
_output_shapes
:
?*?*
dtype0
u
dense_165/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_165/bias
n
"dense_165/bias/Read/ReadVariableOpReadVariableOpdense_165/bias*
_output_shapes	
:?*
dtype0
~
dense_166/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*!
shared_namedense_166/kernel
w
$dense_166/kernel/Read/ReadVariableOpReadVariableOpdense_166/kernel* 
_output_shapes
:
??*
dtype0
u
dense_166/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_166/bias
n
"dense_166/bias/Read/ReadVariableOpReadVariableOpdense_166/bias*
_output_shapes	
:?*
dtype0
}
dense_167/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?
*!
shared_namedense_167/kernel
v
$dense_167/kernel/Read/ReadVariableOpReadVariableOpdense_167/kernel*
_output_shapes
:	?
*
dtype0
t
dense_167/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_167/bias
m
"dense_167/bias/Read/ReadVariableOpReadVariableOpdense_167/bias*
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
?
Adam/conv2d_77/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_77/kernel/m
?
+Adam/conv2d_77/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_77/kernel/m*&
_output_shapes
: *
dtype0
?
Adam/conv2d_77/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_77/bias/m
{
)Adam/conv2d_77/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_77/bias/m*
_output_shapes
: *
dtype0
?
Adam/dense_165/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?*?*(
shared_nameAdam/dense_165/kernel/m
?
+Adam/dense_165/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_165/kernel/m* 
_output_shapes
:
?*?*
dtype0
?
Adam/dense_165/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/dense_165/bias/m
|
)Adam/dense_165/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_165/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_166/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*(
shared_nameAdam/dense_166/kernel/m
?
+Adam/dense_166/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_166/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/dense_166/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/dense_166/bias/m
|
)Adam/dense_166/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_166/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_167/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?
*(
shared_nameAdam/dense_167/kernel/m
?
+Adam/dense_167/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_167/kernel/m*
_output_shapes
:	?
*
dtype0
?
Adam/dense_167/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/dense_167/bias/m
{
)Adam/dense_167/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_167/bias/m*
_output_shapes
:
*
dtype0
?
Adam/conv2d_77/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_77/kernel/v
?
+Adam/conv2d_77/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_77/kernel/v*&
_output_shapes
: *
dtype0
?
Adam/conv2d_77/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_77/bias/v
{
)Adam/conv2d_77/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_77/bias/v*
_output_shapes
: *
dtype0
?
Adam/dense_165/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?*?*(
shared_nameAdam/dense_165/kernel/v
?
+Adam/dense_165/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_165/kernel/v* 
_output_shapes
:
?*?*
dtype0
?
Adam/dense_165/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/dense_165/bias/v
|
)Adam/dense_165/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_165/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_166/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*(
shared_nameAdam/dense_166/kernel/v
?
+Adam/dense_166/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_166/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/dense_166/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/dense_166/bias/v
|
)Adam/dense_166/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_166/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_167/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?
*(
shared_nameAdam/dense_167/kernel/v
?
+Adam/dense_167/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_167/kernel/v*
_output_shapes
:	?
*
dtype0
?
Adam/dense_167/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/dense_167/bias/v
{
)Adam/dense_167/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_167/bias/v*
_output_shapes
:
*
dtype0

NoOpNoOp
?5
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?4
value?4B?4 B?4
?
layer_with_weights-0
layer-0
layer-1
layer-2
layer-3
layer_with_weights-1
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
	optimizer
		variables

trainable_variables
regularization_losses
	keras_api

signatures
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api
h

 kernel
!bias
"	variables
#trainable_variables
$regularization_losses
%	keras_api
h

&kernel
'bias
(	variables
)trainable_variables
*regularization_losses
+	keras_api
h

,kernel
-bias
.	variables
/trainable_variables
0regularization_losses
1	keras_api
?
2iter

3beta_1

4beta_2
	5decay
6learning_ratemjmk ml!mm&mn'mo,mp-mqvrvs vt!vu&vv'vw,vx-vy
8
0
1
 2
!3
&4
'5
,6
-7
8
0
1
 2
!3
&4
'5
,6
-7
 
?
7metrics
		variables
8non_trainable_variables
9layer_metrics

:layers

trainable_variables
;layer_regularization_losses
regularization_losses
 
\Z
VARIABLE_VALUEconv2d_77/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_77/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
<metrics
	variables
=non_trainable_variables

>layers
?layer_metrics
trainable_variables
@layer_regularization_losses
regularization_losses
 
 
 
?
Ametrics
	variables
Bnon_trainable_variables

Clayers
Dlayer_metrics
trainable_variables
Elayer_regularization_losses
regularization_losses
 
 
 
?
Fmetrics
	variables
Gnon_trainable_variables

Hlayers
Ilayer_metrics
trainable_variables
Jlayer_regularization_losses
regularization_losses
 
 
 
?
Kmetrics
	variables
Lnon_trainable_variables

Mlayers
Nlayer_metrics
trainable_variables
Olayer_regularization_losses
regularization_losses
\Z
VARIABLE_VALUEdense_165/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_165/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

 0
!1

 0
!1
 
?
Pmetrics
"	variables
Qnon_trainable_variables

Rlayers
Slayer_metrics
#trainable_variables
Tlayer_regularization_losses
$regularization_losses
\Z
VARIABLE_VALUEdense_166/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_166/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

&0
'1

&0
'1
 
?
Umetrics
(	variables
Vnon_trainable_variables

Wlayers
Xlayer_metrics
)trainable_variables
Ylayer_regularization_losses
*regularization_losses
\Z
VARIABLE_VALUEdense_167/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_167/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

,0
-1

,0
-1
 
?
Zmetrics
.	variables
[non_trainable_variables

\layers
]layer_metrics
/trainable_variables
^layer_regularization_losses
0regularization_losses
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

_0
`1
 
 
1
0
1
2
3
4
5
6
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
4
	atotal
	bcount
c	variables
d	keras_api
D
	etotal
	fcount
g
_fn_kwargs
h	variables
i	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

a0
b1

c	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

e0
f1

h	variables
}
VARIABLE_VALUEAdam/conv2d_77/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_77/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_165/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_165/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_166/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_166/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_167/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_167/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_77/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_77/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_165/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_165/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_166/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_166/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_167/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_167/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_conv2d_77_inputPlaceholder*/
_output_shapes
:?????????*
dtype0*$
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv2d_77_inputconv2d_77/kernelconv2d_77/biasdense_165/kerneldense_165/biasdense_166/kerneldense_166/biasdense_167/kerneldense_167/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *.
f)R'
%__inference_signature_wrapper_2660055
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_77/kernel/Read/ReadVariableOp"conv2d_77/bias/Read/ReadVariableOp$dense_165/kernel/Read/ReadVariableOp"dense_165/bias/Read/ReadVariableOp$dense_166/kernel/Read/ReadVariableOp"dense_166/bias/Read/ReadVariableOp$dense_167/kernel/Read/ReadVariableOp"dense_167/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp+Adam/conv2d_77/kernel/m/Read/ReadVariableOp)Adam/conv2d_77/bias/m/Read/ReadVariableOp+Adam/dense_165/kernel/m/Read/ReadVariableOp)Adam/dense_165/bias/m/Read/ReadVariableOp+Adam/dense_166/kernel/m/Read/ReadVariableOp)Adam/dense_166/bias/m/Read/ReadVariableOp+Adam/dense_167/kernel/m/Read/ReadVariableOp)Adam/dense_167/bias/m/Read/ReadVariableOp+Adam/conv2d_77/kernel/v/Read/ReadVariableOp)Adam/conv2d_77/bias/v/Read/ReadVariableOp+Adam/dense_165/kernel/v/Read/ReadVariableOp)Adam/dense_165/bias/v/Read/ReadVariableOp+Adam/dense_166/kernel/v/Read/ReadVariableOp)Adam/dense_166/bias/v/Read/ReadVariableOp+Adam/dense_167/kernel/v/Read/ReadVariableOp)Adam/dense_167/bias/v/Read/ReadVariableOpConst*.
Tin'
%2#	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference__traced_save_2660416
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_77/kernelconv2d_77/biasdense_165/kerneldense_165/biasdense_166/kerneldense_166/biasdense_167/kerneldense_167/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/conv2d_77/kernel/mAdam/conv2d_77/bias/mAdam/dense_165/kernel/mAdam/dense_165/bias/mAdam/dense_166/kernel/mAdam/dense_166/bias/mAdam/dense_167/kernel/mAdam/dense_167/bias/mAdam/conv2d_77/kernel/vAdam/conv2d_77/bias/vAdam/dense_165/kernel/vAdam/dense_165/bias/vAdam/dense_166/kernel/vAdam/dense_166/bias/vAdam/dense_167/kernel/vAdam/dense_167/bias/v*-
Tin&
$2"*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference__traced_restore_2660525??
?
c
G__inference_flatten_55_layer_call_and_return_conditional_losses_2659810

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????*2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????*2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?7
?
G__inference_Easy_Model_layer_call_and_return_conditional_losses_2660098

inputs,
(conv2d_77_conv2d_readvariableop_resource-
)conv2d_77_biasadd_readvariableop_resource,
(dense_165_matmul_readvariableop_resource-
)dense_165_biasadd_readvariableop_resource,
(dense_166_matmul_readvariableop_resource-
)dense_166_biasadd_readvariableop_resource,
(dense_167_matmul_readvariableop_resource-
)dense_167_biasadd_readvariableop_resource
identity?? conv2d_77/BiasAdd/ReadVariableOp?conv2d_77/Conv2D/ReadVariableOp? dense_165/BiasAdd/ReadVariableOp?dense_165/MatMul/ReadVariableOp? dense_166/BiasAdd/ReadVariableOp?dense_166/MatMul/ReadVariableOp? dense_167/BiasAdd/ReadVariableOp?dense_167/MatMul/ReadVariableOp?
conv2d_77/Conv2D/ReadVariableOpReadVariableOp(conv2d_77_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_77/Conv2D/ReadVariableOp?
conv2d_77/Conv2DConv2Dinputs'conv2d_77/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
conv2d_77/Conv2D?
 conv2d_77/BiasAdd/ReadVariableOpReadVariableOp)conv2d_77_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_77/BiasAdd/ReadVariableOp?
conv2d_77/BiasAddBiasAddconv2d_77/Conv2D:output:0(conv2d_77/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_77/BiasAdd~
conv2d_77/ReluReluconv2d_77/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2d_77/Relu?
max_pooling2d_55/MaxPoolMaxPoolconv2d_77/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2
max_pooling2d_55/MaxPooly
dropout_55/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_55/dropout/Const?
dropout_55/dropout/MulMul!max_pooling2d_55/MaxPool:output:0!dropout_55/dropout/Const:output:0*
T0*/
_output_shapes
:????????? 2
dropout_55/dropout/Mul?
dropout_55/dropout/ShapeShape!max_pooling2d_55/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_55/dropout/Shape?
/dropout_55/dropout/random_uniform/RandomUniformRandomUniform!dropout_55/dropout/Shape:output:0*
T0*/
_output_shapes
:????????? *
dtype021
/dropout_55/dropout/random_uniform/RandomUniform?
!dropout_55/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2#
!dropout_55/dropout/GreaterEqual/y?
dropout_55/dropout/GreaterEqualGreaterEqual8dropout_55/dropout/random_uniform/RandomUniform:output:0*dropout_55/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:????????? 2!
dropout_55/dropout/GreaterEqual?
dropout_55/dropout/CastCast#dropout_55/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:????????? 2
dropout_55/dropout/Cast?
dropout_55/dropout/Mul_1Muldropout_55/dropout/Mul:z:0dropout_55/dropout/Cast:y:0*
T0*/
_output_shapes
:????????? 2
dropout_55/dropout/Mul_1u
flatten_55/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_55/Const?
flatten_55/ReshapeReshapedropout_55/dropout/Mul_1:z:0flatten_55/Const:output:0*
T0*(
_output_shapes
:??????????*2
flatten_55/Reshape?
dense_165/MatMul/ReadVariableOpReadVariableOp(dense_165_matmul_readvariableop_resource* 
_output_shapes
:
?*?*
dtype02!
dense_165/MatMul/ReadVariableOp?
dense_165/MatMulMatMulflatten_55/Reshape:output:0'dense_165/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_165/MatMul?
 dense_165/BiasAdd/ReadVariableOpReadVariableOp)dense_165_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 dense_165/BiasAdd/ReadVariableOp?
dense_165/BiasAddBiasAdddense_165/MatMul:product:0(dense_165/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_165/BiasAddw
dense_165/ReluReludense_165/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_165/Relu?
dense_166/MatMul/ReadVariableOpReadVariableOp(dense_166_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02!
dense_166/MatMul/ReadVariableOp?
dense_166/MatMulMatMuldense_165/Relu:activations:0'dense_166/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_166/MatMul?
 dense_166/BiasAdd/ReadVariableOpReadVariableOp)dense_166_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 dense_166/BiasAdd/ReadVariableOp?
dense_166/BiasAddBiasAdddense_166/MatMul:product:0(dense_166/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_166/BiasAddw
dense_166/ReluReludense_166/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_166/Relu?
dense_167/MatMul/ReadVariableOpReadVariableOp(dense_167_matmul_readvariableop_resource*
_output_shapes
:	?
*
dtype02!
dense_167/MatMul/ReadVariableOp?
dense_167/MatMulMatMuldense_166/Relu:activations:0'dense_167/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_167/MatMul?
 dense_167/BiasAdd/ReadVariableOpReadVariableOp)dense_167_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02"
 dense_167/BiasAdd/ReadVariableOp?
dense_167/BiasAddBiasAdddense_167/MatMul:product:0(dense_167/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_167/BiasAdd
dense_167/SoftmaxSoftmaxdense_167/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
dense_167/Softmax?
IdentityIdentitydense_167/Softmax:softmax:0!^conv2d_77/BiasAdd/ReadVariableOp ^conv2d_77/Conv2D/ReadVariableOp!^dense_165/BiasAdd/ReadVariableOp ^dense_165/MatMul/ReadVariableOp!^dense_166/BiasAdd/ReadVariableOp ^dense_166/MatMul/ReadVariableOp!^dense_167/BiasAdd/ReadVariableOp ^dense_167/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::2D
 conv2d_77/BiasAdd/ReadVariableOp conv2d_77/BiasAdd/ReadVariableOp2B
conv2d_77/Conv2D/ReadVariableOpconv2d_77/Conv2D/ReadVariableOp2D
 dense_165/BiasAdd/ReadVariableOp dense_165/BiasAdd/ReadVariableOp2B
dense_165/MatMul/ReadVariableOpdense_165/MatMul/ReadVariableOp2D
 dense_166/BiasAdd/ReadVariableOp dense_166/BiasAdd/ReadVariableOp2B
dense_166/MatMul/ReadVariableOpdense_166/MatMul/ReadVariableOp2D
 dense_167/BiasAdd/ReadVariableOp dense_167/BiasAdd/ReadVariableOp2B
dense_167/MatMul/ReadVariableOpdense_167/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
%__inference_signature_wrapper_2660055
conv2d_77_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_77_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *+
f&R$
"__inference__wrapped_model_26597302
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:?????????
)
_user_specified_nameconv2d_77_input
?
f
G__inference_dropout_55_layer_call_and_return_conditional_losses_2659786

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:????????? 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:????????? *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:????????? 2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:????????? 2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:????????? 2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?

?
F__inference_conv2d_77_layer_call_and_return_conditional_losses_2660187

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
e
G__inference_dropout_55_layer_call_and_return_conditional_losses_2660213

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:????????? 2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:????????? 2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
? 
?
G__inference_Easy_Model_layer_call_and_return_conditional_losses_2659927
conv2d_77_input
conv2d_77_2659903
conv2d_77_2659905
dense_165_2659911
dense_165_2659913
dense_166_2659916
dense_166_2659918
dense_167_2659921
dense_167_2659923
identity??!conv2d_77/StatefulPartitionedCall?!dense_165/StatefulPartitionedCall?!dense_166/StatefulPartitionedCall?!dense_167/StatefulPartitionedCall?
!conv2d_77/StatefulPartitionedCallStatefulPartitionedCallconv2d_77_inputconv2d_77_2659903conv2d_77_2659905*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_77_layer_call_and_return_conditional_losses_26597572#
!conv2d_77/StatefulPartitionedCall?
 max_pooling2d_55/PartitionedCallPartitionedCall*conv2d_77/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_max_pooling2d_55_layer_call_and_return_conditional_losses_26597362"
 max_pooling2d_55/PartitionedCall?
dropout_55/PartitionedCallPartitionedCall)max_pooling2d_55/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dropout_55_layer_call_and_return_conditional_losses_26597912
dropout_55/PartitionedCall?
flatten_55/PartitionedCallPartitionedCall#dropout_55/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????** 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_flatten_55_layer_call_and_return_conditional_losses_26598102
flatten_55/PartitionedCall?
!dense_165/StatefulPartitionedCallStatefulPartitionedCall#flatten_55/PartitionedCall:output:0dense_165_2659911dense_165_2659913*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_165_layer_call_and_return_conditional_losses_26598292#
!dense_165/StatefulPartitionedCall?
!dense_166/StatefulPartitionedCallStatefulPartitionedCall*dense_165/StatefulPartitionedCall:output:0dense_166_2659916dense_166_2659918*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_166_layer_call_and_return_conditional_losses_26598562#
!dense_166/StatefulPartitionedCall?
!dense_167/StatefulPartitionedCallStatefulPartitionedCall*dense_166/StatefulPartitionedCall:output:0dense_167_2659921dense_167_2659923*
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
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_167_layer_call_and_return_conditional_losses_26598832#
!dense_167/StatefulPartitionedCall?
IdentityIdentity*dense_167/StatefulPartitionedCall:output:0"^conv2d_77/StatefulPartitionedCall"^dense_165/StatefulPartitionedCall"^dense_166/StatefulPartitionedCall"^dense_167/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::2F
!conv2d_77/StatefulPartitionedCall!conv2d_77/StatefulPartitionedCall2F
!dense_165/StatefulPartitionedCall!dense_165/StatefulPartitionedCall2F
!dense_166/StatefulPartitionedCall!dense_166/StatefulPartitionedCall2F
!dense_167/StatefulPartitionedCall!dense_167/StatefulPartitionedCall:` \
/
_output_shapes
:?????????
)
_user_specified_nameconv2d_77_input
?	
?
F__inference_dense_165_layer_call_and_return_conditional_losses_2659829

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
?*?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????*::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????*
 
_user_specified_nameinputs
?-
?
G__inference_Easy_Model_layer_call_and_return_conditional_losses_2660134

inputs,
(conv2d_77_conv2d_readvariableop_resource-
)conv2d_77_biasadd_readvariableop_resource,
(dense_165_matmul_readvariableop_resource-
)dense_165_biasadd_readvariableop_resource,
(dense_166_matmul_readvariableop_resource-
)dense_166_biasadd_readvariableop_resource,
(dense_167_matmul_readvariableop_resource-
)dense_167_biasadd_readvariableop_resource
identity?? conv2d_77/BiasAdd/ReadVariableOp?conv2d_77/Conv2D/ReadVariableOp? dense_165/BiasAdd/ReadVariableOp?dense_165/MatMul/ReadVariableOp? dense_166/BiasAdd/ReadVariableOp?dense_166/MatMul/ReadVariableOp? dense_167/BiasAdd/ReadVariableOp?dense_167/MatMul/ReadVariableOp?
conv2d_77/Conv2D/ReadVariableOpReadVariableOp(conv2d_77_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_77/Conv2D/ReadVariableOp?
conv2d_77/Conv2DConv2Dinputs'conv2d_77/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
conv2d_77/Conv2D?
 conv2d_77/BiasAdd/ReadVariableOpReadVariableOp)conv2d_77_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_77/BiasAdd/ReadVariableOp?
conv2d_77/BiasAddBiasAddconv2d_77/Conv2D:output:0(conv2d_77/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_77/BiasAdd~
conv2d_77/ReluReluconv2d_77/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2d_77/Relu?
max_pooling2d_55/MaxPoolMaxPoolconv2d_77/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2
max_pooling2d_55/MaxPool?
dropout_55/IdentityIdentity!max_pooling2d_55/MaxPool:output:0*
T0*/
_output_shapes
:????????? 2
dropout_55/Identityu
flatten_55/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_55/Const?
flatten_55/ReshapeReshapedropout_55/Identity:output:0flatten_55/Const:output:0*
T0*(
_output_shapes
:??????????*2
flatten_55/Reshape?
dense_165/MatMul/ReadVariableOpReadVariableOp(dense_165_matmul_readvariableop_resource* 
_output_shapes
:
?*?*
dtype02!
dense_165/MatMul/ReadVariableOp?
dense_165/MatMulMatMulflatten_55/Reshape:output:0'dense_165/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_165/MatMul?
 dense_165/BiasAdd/ReadVariableOpReadVariableOp)dense_165_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 dense_165/BiasAdd/ReadVariableOp?
dense_165/BiasAddBiasAdddense_165/MatMul:product:0(dense_165/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_165/BiasAddw
dense_165/ReluReludense_165/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_165/Relu?
dense_166/MatMul/ReadVariableOpReadVariableOp(dense_166_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02!
dense_166/MatMul/ReadVariableOp?
dense_166/MatMulMatMuldense_165/Relu:activations:0'dense_166/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_166/MatMul?
 dense_166/BiasAdd/ReadVariableOpReadVariableOp)dense_166_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 dense_166/BiasAdd/ReadVariableOp?
dense_166/BiasAddBiasAdddense_166/MatMul:product:0(dense_166/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_166/BiasAddw
dense_166/ReluReludense_166/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_166/Relu?
dense_167/MatMul/ReadVariableOpReadVariableOp(dense_167_matmul_readvariableop_resource*
_output_shapes
:	?
*
dtype02!
dense_167/MatMul/ReadVariableOp?
dense_167/MatMulMatMuldense_166/Relu:activations:0'dense_167/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_167/MatMul?
 dense_167/BiasAdd/ReadVariableOpReadVariableOp)dense_167_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02"
 dense_167/BiasAdd/ReadVariableOp?
dense_167/BiasAddBiasAdddense_167/MatMul:product:0(dense_167/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_167/BiasAdd
dense_167/SoftmaxSoftmaxdense_167/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
dense_167/Softmax?
IdentityIdentitydense_167/Softmax:softmax:0!^conv2d_77/BiasAdd/ReadVariableOp ^conv2d_77/Conv2D/ReadVariableOp!^dense_165/BiasAdd/ReadVariableOp ^dense_165/MatMul/ReadVariableOp!^dense_166/BiasAdd/ReadVariableOp ^dense_166/MatMul/ReadVariableOp!^dense_167/BiasAdd/ReadVariableOp ^dense_167/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::2D
 conv2d_77/BiasAdd/ReadVariableOp conv2d_77/BiasAdd/ReadVariableOp2B
conv2d_77/Conv2D/ReadVariableOpconv2d_77/Conv2D/ReadVariableOp2D
 dense_165/BiasAdd/ReadVariableOp dense_165/BiasAdd/ReadVariableOp2B
dense_165/MatMul/ReadVariableOpdense_165/MatMul/ReadVariableOp2D
 dense_166/BiasAdd/ReadVariableOp dense_166/BiasAdd/ReadVariableOp2B
dense_166/MatMul/ReadVariableOpdense_166/MatMul/ReadVariableOp2D
 dense_167/BiasAdd/ReadVariableOp dense_167/BiasAdd/ReadVariableOp2B
dense_167/MatMul/ReadVariableOpdense_167/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
F__inference_dense_167_layer_call_and_return_conditional_losses_2660285

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
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
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
F__inference_conv2d_77_layer_call_and_return_conditional_losses_2659757

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
H
,__inference_dropout_55_layer_call_fn_2660223

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
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dropout_55_layer_call_and_return_conditional_losses_26597912
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
+__inference_dense_166_layer_call_fn_2660274

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_166_layer_call_and_return_conditional_losses_26598562
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?H
?
 __inference__traced_save_2660416
file_prefix/
+savev2_conv2d_77_kernel_read_readvariableop-
)savev2_conv2d_77_bias_read_readvariableop/
+savev2_dense_165_kernel_read_readvariableop-
)savev2_dense_165_bias_read_readvariableop/
+savev2_dense_166_kernel_read_readvariableop-
)savev2_dense_166_bias_read_readvariableop/
+savev2_dense_167_kernel_read_readvariableop-
)savev2_dense_167_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop6
2savev2_adam_conv2d_77_kernel_m_read_readvariableop4
0savev2_adam_conv2d_77_bias_m_read_readvariableop6
2savev2_adam_dense_165_kernel_m_read_readvariableop4
0savev2_adam_dense_165_bias_m_read_readvariableop6
2savev2_adam_dense_166_kernel_m_read_readvariableop4
0savev2_adam_dense_166_bias_m_read_readvariableop6
2savev2_adam_dense_167_kernel_m_read_readvariableop4
0savev2_adam_dense_167_bias_m_read_readvariableop6
2savev2_adam_conv2d_77_kernel_v_read_readvariableop4
0savev2_adam_conv2d_77_bias_v_read_readvariableop6
2savev2_adam_dense_165_kernel_v_read_readvariableop4
0savev2_adam_dense_165_bias_v_read_readvariableop6
2savev2_adam_dense_166_kernel_v_read_readvariableop4
0savev2_adam_dense_166_bias_v_read_readvariableop6
2savev2_adam_dense_167_kernel_v_read_readvariableop4
0savev2_adam_dense_167_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
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
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
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
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*?
value?B?"B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_77_kernel_read_readvariableop)savev2_conv2d_77_bias_read_readvariableop+savev2_dense_165_kernel_read_readvariableop)savev2_dense_165_bias_read_readvariableop+savev2_dense_166_kernel_read_readvariableop)savev2_dense_166_bias_read_readvariableop+savev2_dense_167_kernel_read_readvariableop)savev2_dense_167_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop2savev2_adam_conv2d_77_kernel_m_read_readvariableop0savev2_adam_conv2d_77_bias_m_read_readvariableop2savev2_adam_dense_165_kernel_m_read_readvariableop0savev2_adam_dense_165_bias_m_read_readvariableop2savev2_adam_dense_166_kernel_m_read_readvariableop0savev2_adam_dense_166_bias_m_read_readvariableop2savev2_adam_dense_167_kernel_m_read_readvariableop0savev2_adam_dense_167_bias_m_read_readvariableop2savev2_adam_conv2d_77_kernel_v_read_readvariableop0savev2_adam_conv2d_77_bias_v_read_readvariableop2savev2_adam_dense_165_kernel_v_read_readvariableop0savev2_adam_dense_165_bias_v_read_readvariableop2savev2_adam_dense_166_kernel_v_read_readvariableop0savev2_adam_dense_166_bias_v_read_readvariableop2savev2_adam_dense_167_kernel_v_read_readvariableop0savev2_adam_dense_167_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *0
dtypes&
$2"	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
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

identity_1Identity_1:output:0*?
_input_shapes?
?: : : :
?*?:?:
??:?:	?
:
: : : : : : : : : : : :
?*?:?:
??:?:	?
:
: : :
?*?:?:
??:?:	?
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
: : 

_output_shapes
: :&"
 
_output_shapes
:
?*?:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?
: 

_output_shapes
:
:	

_output_shapes
: :


_output_shapes
: :
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
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
: :&"
 
_output_shapes
:
?*?:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?
: 

_output_shapes
:
:,(
&
_output_shapes
: : 

_output_shapes
: :&"
 
_output_shapes
:
?*?:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:% !

_output_shapes
:	?
: !

_output_shapes
:
:"

_output_shapes
: 
?
?
,__inference_Easy_Model_layer_call_fn_2660024
conv2d_77_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_77_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_Easy_Model_layer_call_and_return_conditional_losses_26600052
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:?????????
)
_user_specified_nameconv2d_77_input
?
e
,__inference_dropout_55_layer_call_fn_2660218

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dropout_55_layer_call_and_return_conditional_losses_26597862
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
+__inference_dense_167_layer_call_fn_2660294

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
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
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_167_layer_call_and_return_conditional_losses_26598832
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
? 
?
G__inference_Easy_Model_layer_call_and_return_conditional_losses_2660005

inputs
conv2d_77_2659981
conv2d_77_2659983
dense_165_2659989
dense_165_2659991
dense_166_2659994
dense_166_2659996
dense_167_2659999
dense_167_2660001
identity??!conv2d_77/StatefulPartitionedCall?!dense_165/StatefulPartitionedCall?!dense_166/StatefulPartitionedCall?!dense_167/StatefulPartitionedCall?
!conv2d_77/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_77_2659981conv2d_77_2659983*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_77_layer_call_and_return_conditional_losses_26597572#
!conv2d_77/StatefulPartitionedCall?
 max_pooling2d_55/PartitionedCallPartitionedCall*conv2d_77/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_max_pooling2d_55_layer_call_and_return_conditional_losses_26597362"
 max_pooling2d_55/PartitionedCall?
dropout_55/PartitionedCallPartitionedCall)max_pooling2d_55/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dropout_55_layer_call_and_return_conditional_losses_26597912
dropout_55/PartitionedCall?
flatten_55/PartitionedCallPartitionedCall#dropout_55/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????** 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_flatten_55_layer_call_and_return_conditional_losses_26598102
flatten_55/PartitionedCall?
!dense_165/StatefulPartitionedCallStatefulPartitionedCall#flatten_55/PartitionedCall:output:0dense_165_2659989dense_165_2659991*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_165_layer_call_and_return_conditional_losses_26598292#
!dense_165/StatefulPartitionedCall?
!dense_166/StatefulPartitionedCallStatefulPartitionedCall*dense_165/StatefulPartitionedCall:output:0dense_166_2659994dense_166_2659996*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_166_layer_call_and_return_conditional_losses_26598562#
!dense_166/StatefulPartitionedCall?
!dense_167/StatefulPartitionedCallStatefulPartitionedCall*dense_166/StatefulPartitionedCall:output:0dense_167_2659999dense_167_2660001*
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
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_167_layer_call_and_return_conditional_losses_26598832#
!dense_167/StatefulPartitionedCall?
IdentityIdentity*dense_167/StatefulPartitionedCall:output:0"^conv2d_77/StatefulPartitionedCall"^dense_165/StatefulPartitionedCall"^dense_166/StatefulPartitionedCall"^dense_167/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::2F
!conv2d_77/StatefulPartitionedCall!conv2d_77/StatefulPartitionedCall2F
!dense_165/StatefulPartitionedCall!dense_165/StatefulPartitionedCall2F
!dense_166/StatefulPartitionedCall!dense_166/StatefulPartitionedCall2F
!dense_167/StatefulPartitionedCall!dense_167/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
+__inference_dense_165_layer_call_fn_2660254

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_165_layer_call_and_return_conditional_losses_26598292
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????*::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????*
 
_user_specified_nameinputs
?
e
G__inference_dropout_55_layer_call_and_return_conditional_losses_2659791

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:????????? 2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:????????? 2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
,__inference_Easy_Model_layer_call_fn_2660155

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_Easy_Model_layer_call_and_return_conditional_losses_26599572
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
F__inference_dense_166_layer_call_and_return_conditional_losses_2659856

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?!
?
G__inference_Easy_Model_layer_call_and_return_conditional_losses_2659957

inputs
conv2d_77_2659933
conv2d_77_2659935
dense_165_2659941
dense_165_2659943
dense_166_2659946
dense_166_2659948
dense_167_2659951
dense_167_2659953
identity??!conv2d_77/StatefulPartitionedCall?!dense_165/StatefulPartitionedCall?!dense_166/StatefulPartitionedCall?!dense_167/StatefulPartitionedCall?"dropout_55/StatefulPartitionedCall?
!conv2d_77/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_77_2659933conv2d_77_2659935*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_77_layer_call_and_return_conditional_losses_26597572#
!conv2d_77/StatefulPartitionedCall?
 max_pooling2d_55/PartitionedCallPartitionedCall*conv2d_77/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_max_pooling2d_55_layer_call_and_return_conditional_losses_26597362"
 max_pooling2d_55/PartitionedCall?
"dropout_55/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_55/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dropout_55_layer_call_and_return_conditional_losses_26597862$
"dropout_55/StatefulPartitionedCall?
flatten_55/PartitionedCallPartitionedCall+dropout_55/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????** 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_flatten_55_layer_call_and_return_conditional_losses_26598102
flatten_55/PartitionedCall?
!dense_165/StatefulPartitionedCallStatefulPartitionedCall#flatten_55/PartitionedCall:output:0dense_165_2659941dense_165_2659943*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_165_layer_call_and_return_conditional_losses_26598292#
!dense_165/StatefulPartitionedCall?
!dense_166/StatefulPartitionedCallStatefulPartitionedCall*dense_165/StatefulPartitionedCall:output:0dense_166_2659946dense_166_2659948*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_166_layer_call_and_return_conditional_losses_26598562#
!dense_166/StatefulPartitionedCall?
!dense_167/StatefulPartitionedCallStatefulPartitionedCall*dense_166/StatefulPartitionedCall:output:0dense_167_2659951dense_167_2659953*
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
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_167_layer_call_and_return_conditional_losses_26598832#
!dense_167/StatefulPartitionedCall?
IdentityIdentity*dense_167/StatefulPartitionedCall:output:0"^conv2d_77/StatefulPartitionedCall"^dense_165/StatefulPartitionedCall"^dense_166/StatefulPartitionedCall"^dense_167/StatefulPartitionedCall#^dropout_55/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::2F
!conv2d_77/StatefulPartitionedCall!conv2d_77/StatefulPartitionedCall2F
!dense_165/StatefulPartitionedCall!dense_165/StatefulPartitionedCall2F
!dense_166/StatefulPartitionedCall!dense_166/StatefulPartitionedCall2F
!dense_167/StatefulPartitionedCall!dense_167/StatefulPartitionedCall2H
"dropout_55/StatefulPartitionedCall"dropout_55/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
N
2__inference_max_pooling2d_55_layer_call_fn_2659742

inputs
identity?
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
 *0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_max_pooling2d_55_layer_call_and_return_conditional_losses_26597362
PartitionedCall?
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
?
f
G__inference_dropout_55_layer_call_and_return_conditional_losses_2660208

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:????????? 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:????????? *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:????????? 2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:????????? 2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:????????? 2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?"
?
G__inference_Easy_Model_layer_call_and_return_conditional_losses_2659900
conv2d_77_input
conv2d_77_2659768
conv2d_77_2659770
dense_165_2659840
dense_165_2659842
dense_166_2659867
dense_166_2659869
dense_167_2659894
dense_167_2659896
identity??!conv2d_77/StatefulPartitionedCall?!dense_165/StatefulPartitionedCall?!dense_166/StatefulPartitionedCall?!dense_167/StatefulPartitionedCall?"dropout_55/StatefulPartitionedCall?
!conv2d_77/StatefulPartitionedCallStatefulPartitionedCallconv2d_77_inputconv2d_77_2659768conv2d_77_2659770*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_77_layer_call_and_return_conditional_losses_26597572#
!conv2d_77/StatefulPartitionedCall?
 max_pooling2d_55/PartitionedCallPartitionedCall*conv2d_77/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_max_pooling2d_55_layer_call_and_return_conditional_losses_26597362"
 max_pooling2d_55/PartitionedCall?
"dropout_55/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_55/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_dropout_55_layer_call_and_return_conditional_losses_26597862$
"dropout_55/StatefulPartitionedCall?
flatten_55/PartitionedCallPartitionedCall+dropout_55/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????** 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_flatten_55_layer_call_and_return_conditional_losses_26598102
flatten_55/PartitionedCall?
!dense_165/StatefulPartitionedCallStatefulPartitionedCall#flatten_55/PartitionedCall:output:0dense_165_2659840dense_165_2659842*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_165_layer_call_and_return_conditional_losses_26598292#
!dense_165/StatefulPartitionedCall?
!dense_166/StatefulPartitionedCallStatefulPartitionedCall*dense_165/StatefulPartitionedCall:output:0dense_166_2659867dense_166_2659869*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_166_layer_call_and_return_conditional_losses_26598562#
!dense_166/StatefulPartitionedCall?
!dense_167/StatefulPartitionedCallStatefulPartitionedCall*dense_166/StatefulPartitionedCall:output:0dense_167_2659894dense_167_2659896*
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
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_167_layer_call_and_return_conditional_losses_26598832#
!dense_167/StatefulPartitionedCall?
IdentityIdentity*dense_167/StatefulPartitionedCall:output:0"^conv2d_77/StatefulPartitionedCall"^dense_165/StatefulPartitionedCall"^dense_166/StatefulPartitionedCall"^dense_167/StatefulPartitionedCall#^dropout_55/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::2F
!conv2d_77/StatefulPartitionedCall!conv2d_77/StatefulPartitionedCall2F
!dense_165/StatefulPartitionedCall!dense_165/StatefulPartitionedCall2F
!dense_166/StatefulPartitionedCall!dense_166/StatefulPartitionedCall2F
!dense_167/StatefulPartitionedCall!dense_167/StatefulPartitionedCall2H
"dropout_55/StatefulPartitionedCall"dropout_55/StatefulPartitionedCall:` \
/
_output_shapes
:?????????
)
_user_specified_nameconv2d_77_input
?7
?
"__inference__wrapped_model_2659730
conv2d_77_input7
3easy_model_conv2d_77_conv2d_readvariableop_resource8
4easy_model_conv2d_77_biasadd_readvariableop_resource7
3easy_model_dense_165_matmul_readvariableop_resource8
4easy_model_dense_165_biasadd_readvariableop_resource7
3easy_model_dense_166_matmul_readvariableop_resource8
4easy_model_dense_166_biasadd_readvariableop_resource7
3easy_model_dense_167_matmul_readvariableop_resource8
4easy_model_dense_167_biasadd_readvariableop_resource
identity??+Easy_Model/conv2d_77/BiasAdd/ReadVariableOp?*Easy_Model/conv2d_77/Conv2D/ReadVariableOp?+Easy_Model/dense_165/BiasAdd/ReadVariableOp?*Easy_Model/dense_165/MatMul/ReadVariableOp?+Easy_Model/dense_166/BiasAdd/ReadVariableOp?*Easy_Model/dense_166/MatMul/ReadVariableOp?+Easy_Model/dense_167/BiasAdd/ReadVariableOp?*Easy_Model/dense_167/MatMul/ReadVariableOp?
*Easy_Model/conv2d_77/Conv2D/ReadVariableOpReadVariableOp3easy_model_conv2d_77_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02,
*Easy_Model/conv2d_77/Conv2D/ReadVariableOp?
Easy_Model/conv2d_77/Conv2DConv2Dconv2d_77_input2Easy_Model/conv2d_77/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
Easy_Model/conv2d_77/Conv2D?
+Easy_Model/conv2d_77/BiasAdd/ReadVariableOpReadVariableOp4easy_model_conv2d_77_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02-
+Easy_Model/conv2d_77/BiasAdd/ReadVariableOp?
Easy_Model/conv2d_77/BiasAddBiasAdd$Easy_Model/conv2d_77/Conv2D:output:03Easy_Model/conv2d_77/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
Easy_Model/conv2d_77/BiasAdd?
Easy_Model/conv2d_77/ReluRelu%Easy_Model/conv2d_77/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
Easy_Model/conv2d_77/Relu?
#Easy_Model/max_pooling2d_55/MaxPoolMaxPool'Easy_Model/conv2d_77/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2%
#Easy_Model/max_pooling2d_55/MaxPool?
Easy_Model/dropout_55/IdentityIdentity,Easy_Model/max_pooling2d_55/MaxPool:output:0*
T0*/
_output_shapes
:????????? 2 
Easy_Model/dropout_55/Identity?
Easy_Model/flatten_55/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Easy_Model/flatten_55/Const?
Easy_Model/flatten_55/ReshapeReshape'Easy_Model/dropout_55/Identity:output:0$Easy_Model/flatten_55/Const:output:0*
T0*(
_output_shapes
:??????????*2
Easy_Model/flatten_55/Reshape?
*Easy_Model/dense_165/MatMul/ReadVariableOpReadVariableOp3easy_model_dense_165_matmul_readvariableop_resource* 
_output_shapes
:
?*?*
dtype02,
*Easy_Model/dense_165/MatMul/ReadVariableOp?
Easy_Model/dense_165/MatMulMatMul&Easy_Model/flatten_55/Reshape:output:02Easy_Model/dense_165/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Easy_Model/dense_165/MatMul?
+Easy_Model/dense_165/BiasAdd/ReadVariableOpReadVariableOp4easy_model_dense_165_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+Easy_Model/dense_165/BiasAdd/ReadVariableOp?
Easy_Model/dense_165/BiasAddBiasAdd%Easy_Model/dense_165/MatMul:product:03Easy_Model/dense_165/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Easy_Model/dense_165/BiasAdd?
Easy_Model/dense_165/ReluRelu%Easy_Model/dense_165/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Easy_Model/dense_165/Relu?
*Easy_Model/dense_166/MatMul/ReadVariableOpReadVariableOp3easy_model_dense_166_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02,
*Easy_Model/dense_166/MatMul/ReadVariableOp?
Easy_Model/dense_166/MatMulMatMul'Easy_Model/dense_165/Relu:activations:02Easy_Model/dense_166/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Easy_Model/dense_166/MatMul?
+Easy_Model/dense_166/BiasAdd/ReadVariableOpReadVariableOp4easy_model_dense_166_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+Easy_Model/dense_166/BiasAdd/ReadVariableOp?
Easy_Model/dense_166/BiasAddBiasAdd%Easy_Model/dense_166/MatMul:product:03Easy_Model/dense_166/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Easy_Model/dense_166/BiasAdd?
Easy_Model/dense_166/ReluRelu%Easy_Model/dense_166/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Easy_Model/dense_166/Relu?
*Easy_Model/dense_167/MatMul/ReadVariableOpReadVariableOp3easy_model_dense_167_matmul_readvariableop_resource*
_output_shapes
:	?
*
dtype02,
*Easy_Model/dense_167/MatMul/ReadVariableOp?
Easy_Model/dense_167/MatMulMatMul'Easy_Model/dense_166/Relu:activations:02Easy_Model/dense_167/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
Easy_Model/dense_167/MatMul?
+Easy_Model/dense_167/BiasAdd/ReadVariableOpReadVariableOp4easy_model_dense_167_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02-
+Easy_Model/dense_167/BiasAdd/ReadVariableOp?
Easy_Model/dense_167/BiasAddBiasAdd%Easy_Model/dense_167/MatMul:product:03Easy_Model/dense_167/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
Easy_Model/dense_167/BiasAdd?
Easy_Model/dense_167/SoftmaxSoftmax%Easy_Model/dense_167/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
Easy_Model/dense_167/Softmax?
IdentityIdentity&Easy_Model/dense_167/Softmax:softmax:0,^Easy_Model/conv2d_77/BiasAdd/ReadVariableOp+^Easy_Model/conv2d_77/Conv2D/ReadVariableOp,^Easy_Model/dense_165/BiasAdd/ReadVariableOp+^Easy_Model/dense_165/MatMul/ReadVariableOp,^Easy_Model/dense_166/BiasAdd/ReadVariableOp+^Easy_Model/dense_166/MatMul/ReadVariableOp,^Easy_Model/dense_167/BiasAdd/ReadVariableOp+^Easy_Model/dense_167/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::2Z
+Easy_Model/conv2d_77/BiasAdd/ReadVariableOp+Easy_Model/conv2d_77/BiasAdd/ReadVariableOp2X
*Easy_Model/conv2d_77/Conv2D/ReadVariableOp*Easy_Model/conv2d_77/Conv2D/ReadVariableOp2Z
+Easy_Model/dense_165/BiasAdd/ReadVariableOp+Easy_Model/dense_165/BiasAdd/ReadVariableOp2X
*Easy_Model/dense_165/MatMul/ReadVariableOp*Easy_Model/dense_165/MatMul/ReadVariableOp2Z
+Easy_Model/dense_166/BiasAdd/ReadVariableOp+Easy_Model/dense_166/BiasAdd/ReadVariableOp2X
*Easy_Model/dense_166/MatMul/ReadVariableOp*Easy_Model/dense_166/MatMul/ReadVariableOp2Z
+Easy_Model/dense_167/BiasAdd/ReadVariableOp+Easy_Model/dense_167/BiasAdd/ReadVariableOp2X
*Easy_Model/dense_167/MatMul/ReadVariableOp*Easy_Model/dense_167/MatMul/ReadVariableOp:` \
/
_output_shapes
:?????????
)
_user_specified_nameconv2d_77_input
?	
?
F__inference_dense_165_layer_call_and_return_conditional_losses_2660245

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
?*?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????*::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????*
 
_user_specified_nameinputs
?
H
,__inference_flatten_55_layer_call_fn_2660234

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????** 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_flatten_55_layer_call_and_return_conditional_losses_26598102
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????*2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
,__inference_Easy_Model_layer_call_fn_2660176

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_Easy_Model_layer_call_and_return_conditional_losses_26600052
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
i
M__inference_max_pooling2d_55_layer_call_and_return_conditional_losses_2659736

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
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
??
?
#__inference__traced_restore_2660525
file_prefix%
!assignvariableop_conv2d_77_kernel%
!assignvariableop_1_conv2d_77_bias'
#assignvariableop_2_dense_165_kernel%
!assignvariableop_3_dense_165_bias'
#assignvariableop_4_dense_166_kernel%
!assignvariableop_5_dense_166_bias'
#assignvariableop_6_dense_167_kernel%
!assignvariableop_7_dense_167_bias 
assignvariableop_8_adam_iter"
assignvariableop_9_adam_beta_1#
assignvariableop_10_adam_beta_2"
assignvariableop_11_adam_decay*
&assignvariableop_12_adam_learning_rate
assignvariableop_13_total
assignvariableop_14_count
assignvariableop_15_total_1
assignvariableop_16_count_1/
+assignvariableop_17_adam_conv2d_77_kernel_m-
)assignvariableop_18_adam_conv2d_77_bias_m/
+assignvariableop_19_adam_dense_165_kernel_m-
)assignvariableop_20_adam_dense_165_bias_m/
+assignvariableop_21_adam_dense_166_kernel_m-
)assignvariableop_22_adam_dense_166_bias_m/
+assignvariableop_23_adam_dense_167_kernel_m-
)assignvariableop_24_adam_dense_167_bias_m/
+assignvariableop_25_adam_conv2d_77_kernel_v-
)assignvariableop_26_adam_conv2d_77_bias_v/
+assignvariableop_27_adam_dense_165_kernel_v-
)assignvariableop_28_adam_dense_165_bias_v/
+assignvariableop_29_adam_dense_166_kernel_v-
)assignvariableop_30_adam_dense_166_bias_v/
+assignvariableop_31_adam_dense_167_kernel_v-
)assignvariableop_32_adam_dense_167_bias_v
identity_34??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*?
value?B?"B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::*0
dtypes&
$2"	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_77_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_77_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp#assignvariableop_2_dense_165_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp!assignvariableop_3_dense_165_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp#assignvariableop_4_dense_166_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp!assignvariableop_5_dense_166_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp#assignvariableop_6_dense_167_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp!assignvariableop_7_dense_167_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_beta_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp&assignvariableop_12_adam_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_total_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpassignvariableop_16_count_1Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp+assignvariableop_17_adam_conv2d_77_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp)assignvariableop_18_adam_conv2d_77_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp+assignvariableop_19_adam_dense_165_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp)assignvariableop_20_adam_dense_165_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp+assignvariableop_21_adam_dense_166_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp)assignvariableop_22_adam_dense_166_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp+assignvariableop_23_adam_dense_167_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp)assignvariableop_24_adam_dense_167_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp+assignvariableop_25_adam_conv2d_77_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp)assignvariableop_26_adam_conv2d_77_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp+assignvariableop_27_adam_dense_165_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp)assignvariableop_28_adam_dense_165_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp+assignvariableop_29_adam_dense_166_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp)assignvariableop_30_adam_dense_166_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp+assignvariableop_31_adam_dense_167_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp)assignvariableop_32_adam_dense_167_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_329
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_33Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_33?
Identity_34IdentityIdentity_33:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_34"#
identity_34Identity_34:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_32AssignVariableOp_322(
AssignVariableOp_4AssignVariableOp_42(
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
?
?
+__inference_conv2d_77_layer_call_fn_2660196

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_77_layer_call_and_return_conditional_losses_26597572
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
F__inference_dense_166_layer_call_and_return_conditional_losses_2660265

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
,__inference_Easy_Model_layer_call_fn_2659976
conv2d_77_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_77_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_Easy_Model_layer_call_and_return_conditional_losses_26599572
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:?????????
)
_user_specified_nameconv2d_77_input
?	
?
F__inference_dense_167_layer_call_and_return_conditional_losses_2659883

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
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
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
c
G__inference_flatten_55_layer_call_and_return_conditional_losses_2660229

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????*2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????*2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
S
conv2d_77_input@
!serving_default_conv2d_77_input:0?????????=
	dense_1670
StatefulPartitionedCall:0?????????
tensorflow/serving/predict:??
?7
layer_with_weights-0
layer-0
layer-1
layer-2
layer-3
layer_with_weights-1
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
	optimizer
		variables

trainable_variables
regularization_losses
	keras_api

signatures
z_default_save_signature
{__call__
*|&call_and_return_all_conditional_losses"?4
_tf_keras_sequential?4{"class_name": "Sequential", "name": "Easy_Model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "Easy_Model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_77_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_77", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_55", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_55", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten_55", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_165", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_166", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_167", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "Easy_Model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_77_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_77", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_55", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_55", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten_55", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_165", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_166", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_167", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "sparse_categorical_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.009999999776482582, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?


kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
}__call__
*~&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"class_name": "Conv2D", "name": "conv2d_77", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_77", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 1]}}
?
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_55", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_55", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_55", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_55", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
?
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_55", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_55", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?

 kernel
!bias
"	variables
#trainable_variables
$regularization_losses
%	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_165", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_165", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 5408}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 5408]}}
?

&kernel
'bias
(	variables
)trainable_variables
*regularization_losses
+	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_166", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_166", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
?

,kernel
-bias
.	variables
/trainable_variables
0regularization_losses
1	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_167", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_167", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
?
2iter

3beta_1

4beta_2
	5decay
6learning_ratemjmk ml!mm&mn'mo,mp-mqvrvs vt!vu&vv'vw,vx-vy"
	optimizer
X
0
1
 2
!3
&4
'5
,6
-7"
trackable_list_wrapper
X
0
1
 2
!3
&4
'5
,6
-7"
trackable_list_wrapper
 "
trackable_list_wrapper
?
7metrics
		variables
8non_trainable_variables
9layer_metrics

:layers

trainable_variables
;layer_regularization_losses
regularization_losses
{__call__
z_default_save_signature
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
*:( 2conv2d_77/kernel
: 2conv2d_77/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
<metrics
	variables
=non_trainable_variables

>layers
?layer_metrics
trainable_variables
@layer_regularization_losses
regularization_losses
}__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Ametrics
	variables
Bnon_trainable_variables

Clayers
Dlayer_metrics
trainable_variables
Elayer_regularization_losses
regularization_losses
__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Fmetrics
	variables
Gnon_trainable_variables

Hlayers
Ilayer_metrics
trainable_variables
Jlayer_regularization_losses
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Kmetrics
	variables
Lnon_trainable_variables

Mlayers
Nlayer_metrics
trainable_variables
Olayer_regularization_losses
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
$:"
?*?2dense_165/kernel
:?2dense_165/bias
.
 0
!1"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Pmetrics
"	variables
Qnon_trainable_variables

Rlayers
Slayer_metrics
#trainable_variables
Tlayer_regularization_losses
$regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
$:"
??2dense_166/kernel
:?2dense_166/bias
.
&0
'1"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Umetrics
(	variables
Vnon_trainable_variables

Wlayers
Xlayer_metrics
)trainable_variables
Ylayer_regularization_losses
*regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
#:!	?
2dense_167/kernel
:
2dense_167/bias
.
,0
-1"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Zmetrics
.	variables
[non_trainable_variables

\layers
]layer_metrics
/trainable_variables
^layer_regularization_losses
0regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
.
_0
`1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
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
?
	atotal
	bcount
c	variables
d	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
?
	etotal
	fcount
g
_fn_kwargs
h	variables
i	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}}
:  (2total
:  (2count
.
a0
b1"
trackable_list_wrapper
-
c	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
e0
f1"
trackable_list_wrapper
-
h	variables"
_generic_user_object
/:- 2Adam/conv2d_77/kernel/m
!: 2Adam/conv2d_77/bias/m
):'
?*?2Adam/dense_165/kernel/m
": ?2Adam/dense_165/bias/m
):'
??2Adam/dense_166/kernel/m
": ?2Adam/dense_166/bias/m
(:&	?
2Adam/dense_167/kernel/m
!:
2Adam/dense_167/bias/m
/:- 2Adam/conv2d_77/kernel/v
!: 2Adam/conv2d_77/bias/v
):'
?*?2Adam/dense_165/kernel/v
": ?2Adam/dense_165/bias/v
):'
??2Adam/dense_166/kernel/v
": ?2Adam/dense_166/bias/v
(:&	?
2Adam/dense_167/kernel/v
!:
2Adam/dense_167/bias/v
?2?
"__inference__wrapped_model_2659730?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *6?3
1?.
conv2d_77_input?????????
?2?
,__inference_Easy_Model_layer_call_fn_2659976
,__inference_Easy_Model_layer_call_fn_2660176
,__inference_Easy_Model_layer_call_fn_2660155
,__inference_Easy_Model_layer_call_fn_2660024?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_Easy_Model_layer_call_and_return_conditional_losses_2659900
G__inference_Easy_Model_layer_call_and_return_conditional_losses_2660098
G__inference_Easy_Model_layer_call_and_return_conditional_losses_2660134
G__inference_Easy_Model_layer_call_and_return_conditional_losses_2659927?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
+__inference_conv2d_77_layer_call_fn_2660196?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_conv2d_77_layer_call_and_return_conditional_losses_2660187?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
2__inference_max_pooling2d_55_layer_call_fn_2659742?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
M__inference_max_pooling2d_55_layer_call_and_return_conditional_losses_2659736?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
,__inference_dropout_55_layer_call_fn_2660223
,__inference_dropout_55_layer_call_fn_2660218?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_dropout_55_layer_call_and_return_conditional_losses_2660208
G__inference_dropout_55_layer_call_and_return_conditional_losses_2660213?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
,__inference_flatten_55_layer_call_fn_2660234?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_flatten_55_layer_call_and_return_conditional_losses_2660229?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_dense_165_layer_call_fn_2660254?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_dense_165_layer_call_and_return_conditional_losses_2660245?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_dense_166_layer_call_fn_2660274?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_dense_166_layer_call_and_return_conditional_losses_2660265?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_dense_167_layer_call_fn_2660294?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_dense_167_layer_call_and_return_conditional_losses_2660285?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
%__inference_signature_wrapper_2660055conv2d_77_input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
G__inference_Easy_Model_layer_call_and_return_conditional_losses_2659900{ !&',-H?E
>?;
1?.
conv2d_77_input?????????
p

 
? "%?"
?
0?????????

? ?
G__inference_Easy_Model_layer_call_and_return_conditional_losses_2659927{ !&',-H?E
>?;
1?.
conv2d_77_input?????????
p 

 
? "%?"
?
0?????????

? ?
G__inference_Easy_Model_layer_call_and_return_conditional_losses_2660098r !&',-??<
5?2
(?%
inputs?????????
p

 
? "%?"
?
0?????????

? ?
G__inference_Easy_Model_layer_call_and_return_conditional_losses_2660134r !&',-??<
5?2
(?%
inputs?????????
p 

 
? "%?"
?
0?????????

? ?
,__inference_Easy_Model_layer_call_fn_2659976n !&',-H?E
>?;
1?.
conv2d_77_input?????????
p

 
? "??????????
?
,__inference_Easy_Model_layer_call_fn_2660024n !&',-H?E
>?;
1?.
conv2d_77_input?????????
p 

 
? "??????????
?
,__inference_Easy_Model_layer_call_fn_2660155e !&',-??<
5?2
(?%
inputs?????????
p

 
? "??????????
?
,__inference_Easy_Model_layer_call_fn_2660176e !&',-??<
5?2
(?%
inputs?????????
p 

 
? "??????????
?
"__inference__wrapped_model_2659730? !&',-@?=
6?3
1?.
conv2d_77_input?????????
? "5?2
0
	dense_167#? 
	dense_167?????????
?
F__inference_conv2d_77_layer_call_and_return_conditional_losses_2660187l7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0????????? 
? ?
+__inference_conv2d_77_layer_call_fn_2660196_7?4
-?*
(?%
inputs?????????
? " ?????????? ?
F__inference_dense_165_layer_call_and_return_conditional_losses_2660245^ !0?-
&?#
!?
inputs??????????*
? "&?#
?
0??????????
? ?
+__inference_dense_165_layer_call_fn_2660254Q !0?-
&?#
!?
inputs??????????*
? "????????????
F__inference_dense_166_layer_call_and_return_conditional_losses_2660265^&'0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
+__inference_dense_166_layer_call_fn_2660274Q&'0?-
&?#
!?
inputs??????????
? "????????????
F__inference_dense_167_layer_call_and_return_conditional_losses_2660285],-0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????

? 
+__inference_dense_167_layer_call_fn_2660294P,-0?-
&?#
!?
inputs??????????
? "??????????
?
G__inference_dropout_55_layer_call_and_return_conditional_losses_2660208l;?8
1?.
(?%
inputs????????? 
p
? "-?*
#? 
0????????? 
? ?
G__inference_dropout_55_layer_call_and_return_conditional_losses_2660213l;?8
1?.
(?%
inputs????????? 
p 
? "-?*
#? 
0????????? 
? ?
,__inference_dropout_55_layer_call_fn_2660218_;?8
1?.
(?%
inputs????????? 
p
? " ?????????? ?
,__inference_dropout_55_layer_call_fn_2660223_;?8
1?.
(?%
inputs????????? 
p 
? " ?????????? ?
G__inference_flatten_55_layer_call_and_return_conditional_losses_2660229a7?4
-?*
(?%
inputs????????? 
? "&?#
?
0??????????*
? ?
,__inference_flatten_55_layer_call_fn_2660234T7?4
-?*
(?%
inputs????????? 
? "???????????*?
M__inference_max_pooling2d_55_layer_call_and_return_conditional_losses_2659736?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
2__inference_max_pooling2d_55_layer_call_fn_2659742?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
%__inference_signature_wrapper_2660055? !&',-S?P
? 
I?F
D
conv2d_77_input1?.
conv2d_77_input?????????"5?2
0
	dense_167#? 
	dense_167?????????
