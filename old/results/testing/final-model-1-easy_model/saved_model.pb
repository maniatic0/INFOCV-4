??
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
 ?"serve*2.4.12v2.4.0-49-g85c8b2a817f8??
?
conv2d_143/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameconv2d_143/kernel

%conv2d_143/kernel/Read/ReadVariableOpReadVariableOpconv2d_143/kernel*&
_output_shapes
: *
dtype0
v
conv2d_143/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_143/bias
o
#conv2d_143/bias/Read/ReadVariableOpReadVariableOpconv2d_143/bias*
_output_shapes
: *
dtype0
~
dense_308/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?*?*!
shared_namedense_308/kernel
w
$dense_308/kernel/Read/ReadVariableOpReadVariableOpdense_308/kernel* 
_output_shapes
:
?*?*
dtype0
u
dense_308/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_308/bias
n
"dense_308/bias/Read/ReadVariableOpReadVariableOpdense_308/bias*
_output_shapes	
:?*
dtype0
~
dense_309/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*!
shared_namedense_309/kernel
w
$dense_309/kernel/Read/ReadVariableOpReadVariableOpdense_309/kernel* 
_output_shapes
:
??*
dtype0
u
dense_309/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_309/bias
n
"dense_309/bias/Read/ReadVariableOpReadVariableOpdense_309/bias*
_output_shapes	
:?*
dtype0
}
dense_310/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?
*!
shared_namedense_310/kernel
v
$dense_310/kernel/Read/ReadVariableOpReadVariableOpdense_310/kernel*
_output_shapes
:	?
*
dtype0
t
dense_310/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_310/bias
m
"dense_310/bias/Read/ReadVariableOpReadVariableOpdense_310/bias*
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
?
Adam/conv2d_143/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameAdam/conv2d_143/kernel/m
?
,Adam/conv2d_143/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_143/kernel/m*&
_output_shapes
: *
dtype0
?
Adam/conv2d_143/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv2d_143/bias/m
}
*Adam/conv2d_143/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_143/bias/m*
_output_shapes
: *
dtype0
?
Adam/dense_308/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?*?*(
shared_nameAdam/dense_308/kernel/m
?
+Adam/dense_308/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_308/kernel/m* 
_output_shapes
:
?*?*
dtype0
?
Adam/dense_308/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/dense_308/bias/m
|
)Adam/dense_308/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_308/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_309/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*(
shared_nameAdam/dense_309/kernel/m
?
+Adam/dense_309/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_309/kernel/m* 
_output_shapes
:
??*
dtype0
?
Adam/dense_309/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/dense_309/bias/m
|
)Adam/dense_309/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_309/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_310/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?
*(
shared_nameAdam/dense_310/kernel/m
?
+Adam/dense_310/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_310/kernel/m*
_output_shapes
:	?
*
dtype0
?
Adam/dense_310/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/dense_310/bias/m
{
)Adam/dense_310/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_310/bias/m*
_output_shapes
:
*
dtype0
?
Adam/conv2d_143/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameAdam/conv2d_143/kernel/v
?
,Adam/conv2d_143/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_143/kernel/v*&
_output_shapes
: *
dtype0
?
Adam/conv2d_143/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv2d_143/bias/v
}
*Adam/conv2d_143/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_143/bias/v*
_output_shapes
: *
dtype0
?
Adam/dense_308/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?*?*(
shared_nameAdam/dense_308/kernel/v
?
+Adam/dense_308/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_308/kernel/v* 
_output_shapes
:
?*?*
dtype0
?
Adam/dense_308/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/dense_308/bias/v
|
)Adam/dense_308/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_308/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_309/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*(
shared_nameAdam/dense_309/kernel/v
?
+Adam/dense_309/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_309/kernel/v* 
_output_shapes
:
??*
dtype0
?
Adam/dense_309/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/dense_309/bias/v
|
)Adam/dense_309/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_309/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_310/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?
*(
shared_nameAdam/dense_310/kernel/v
?
+Adam/dense_310/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_310/kernel/v*
_output_shapes
:	?
*
dtype0
?
Adam/dense_310/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/dense_310/bias/v
{
)Adam/dense_310/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_310/bias/v*
_output_shapes
:
*
dtype0

NoOpNoOp
?:
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?9
value?9B?9 B?9
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer-4
layer_with_weights-1
layer-5
layer_with_weights-2
layer-6
layer_with_weights-3
layer-7
		optimizer

	variables
regularization_losses
trainable_variables
	keras_api

signatures
y
layer-0
layer-1
layer-2
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
R
	variables
regularization_losses
trainable_variables
	keras_api
R
 	variables
!regularization_losses
"trainable_variables
#	keras_api
R
$	variables
%regularization_losses
&trainable_variables
'	keras_api
h

(kernel
)bias
*	variables
+regularization_losses
,trainable_variables
-	keras_api
h

.kernel
/bias
0	variables
1regularization_losses
2trainable_variables
3	keras_api
h

4kernel
5bias
6	variables
7regularization_losses
8trainable_variables
9	keras_api
?
:iter

;beta_1

<beta_2
	=decay
>learning_ratem~m(m?)m?.m?/m?4m?5m?v?v?(v?)v?.v?/v?4v?5v?
8
0
1
(2
)3
.4
/5
46
57
 
8
0
1
(2
)3
.4
/5
46
57
?
?layer_metrics

	variables
regularization_losses

@layers
Alayer_regularization_losses
Bnon_trainable_variables
Cmetrics
trainable_variables
 

D_rng
E	keras_api

F_rng
G	keras_api

H	keras_api
 
 
 
?
Ilayer_metrics
	variables
regularization_losses

Jlayers
Klayer_regularization_losses
Lnon_trainable_variables
Mmetrics
trainable_variables
][
VARIABLE_VALUEconv2d_143/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_143/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
Nlayer_metrics
	variables
regularization_losses

Olayers
Player_regularization_losses
Qnon_trainable_variables
Rmetrics
trainable_variables
 
 
 
?
Slayer_metrics
	variables
regularization_losses

Tlayers
Ulayer_regularization_losses
Vnon_trainable_variables
Wmetrics
trainable_variables
 
 
 
?
Xlayer_metrics
 	variables
!regularization_losses

Ylayers
Zlayer_regularization_losses
[non_trainable_variables
\metrics
"trainable_variables
 
 
 
?
]layer_metrics
$	variables
%regularization_losses

^layers
_layer_regularization_losses
`non_trainable_variables
ametrics
&trainable_variables
\Z
VARIABLE_VALUEdense_308/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_308/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

(0
)1
 

(0
)1
?
blayer_metrics
*	variables
+regularization_losses

clayers
dlayer_regularization_losses
enon_trainable_variables
fmetrics
,trainable_variables
\Z
VARIABLE_VALUEdense_309/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_309/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

.0
/1
 

.0
/1
?
glayer_metrics
0	variables
1regularization_losses

hlayers
ilayer_regularization_losses
jnon_trainable_variables
kmetrics
2trainable_variables
\Z
VARIABLE_VALUEdense_310/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_310/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

40
51
 

40
51
?
llayer_metrics
6	variables
7regularization_losses

mlayers
nlayer_regularization_losses
onon_trainable_variables
pmetrics
8trainable_variables
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
 
8
0
1
2
3
4
5
6
7
 
 

q0
r1

s
_state_var
 

t
_state_var
 
 
 

0
1
2
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
4
	utotal
	vcount
w	variables
x	keras_api
D
	ytotal
	zcount
{
_fn_kwargs
|	variables
}	keras_api
XV
VARIABLE_VALUEVariable:layer-0/layer-0/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUE
Variable_1:layer-0/layer-1/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

u0
v1

w	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

y0
z1

|	variables
?~
VARIABLE_VALUEAdam/conv2d_143/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_143/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_308/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_308/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_309/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_309/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_310/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_310/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEAdam/conv2d_143/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_143/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_308/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_308/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_309/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_309/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_310/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_310/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
0serving_default_Training_Data_Augmentation_inputPlaceholder*/
_output_shapes
:?????????*
dtype0*$
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCall0serving_default_Training_Data_Augmentation_inputconv2d_143/kernelconv2d_143/biasdense_308/kerneldense_308/biasdense_309/kerneldense_309/biasdense_310/kerneldense_310/bias*
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

*-
config_proto

CPU

GPU 2J 8? *.
f)R'
%__inference_signature_wrapper_2512218
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename%conv2d_143/kernel/Read/ReadVariableOp#conv2d_143/bias/Read/ReadVariableOp$dense_308/kernel/Read/ReadVariableOp"dense_308/bias/Read/ReadVariableOp$dense_309/kernel/Read/ReadVariableOp"dense_309/bias/Read/ReadVariableOp$dense_310/kernel/Read/ReadVariableOp"dense_310/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOpVariable/Read/ReadVariableOpVariable_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp,Adam/conv2d_143/kernel/m/Read/ReadVariableOp*Adam/conv2d_143/bias/m/Read/ReadVariableOp+Adam/dense_308/kernel/m/Read/ReadVariableOp)Adam/dense_308/bias/m/Read/ReadVariableOp+Adam/dense_309/kernel/m/Read/ReadVariableOp)Adam/dense_309/bias/m/Read/ReadVariableOp+Adam/dense_310/kernel/m/Read/ReadVariableOp)Adam/dense_310/bias/m/Read/ReadVariableOp,Adam/conv2d_143/kernel/v/Read/ReadVariableOp*Adam/conv2d_143/bias/v/Read/ReadVariableOp+Adam/dense_308/kernel/v/Read/ReadVariableOp)Adam/dense_308/bias/v/Read/ReadVariableOp+Adam/dense_309/kernel/v/Read/ReadVariableOp)Adam/dense_309/bias/v/Read/ReadVariableOp+Adam/dense_310/kernel/v/Read/ReadVariableOp)Adam/dense_310/bias/v/Read/ReadVariableOpConst*0
Tin)
'2%			*
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
GPU 2J 8? *)
f$R"
 __inference__traced_save_2512889
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_143/kernelconv2d_143/biasdense_308/kerneldense_308/biasdense_309/kerneldense_309/biasdense_310/kerneldense_310/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateVariable
Variable_1totalcounttotal_1count_1Adam/conv2d_143/kernel/mAdam/conv2d_143/bias/mAdam/dense_308/kernel/mAdam/dense_308/bias/mAdam/dense_309/kernel/mAdam/dense_309/bias/mAdam/dense_310/kernel/mAdam/dense_310/bias/mAdam/conv2d_143/kernel/vAdam/conv2d_143/bias/vAdam/dense_308/kernel/vAdam/dense_308/bias/vAdam/dense_309/kernel/vAdam/dense_309/bias/vAdam/dense_310/kernel/vAdam/dense_310/bias/v*/
Tin(
&2$*
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
GPU 2J 8? *,
f'R%
#__inference__traced_restore_2513004ȥ
?	
?
F__inference_dense_308_layer_call_and_return_conditional_losses_2511985

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
??
?
#__inference__traced_restore_2513004
file_prefix&
"assignvariableop_conv2d_143_kernel&
"assignvariableop_1_conv2d_143_bias'
#assignvariableop_2_dense_308_kernel%
!assignvariableop_3_dense_308_bias'
#assignvariableop_4_dense_309_kernel%
!assignvariableop_5_dense_309_bias'
#assignvariableop_6_dense_310_kernel%
!assignvariableop_7_dense_310_bias 
assignvariableop_8_adam_iter"
assignvariableop_9_adam_beta_1#
assignvariableop_10_adam_beta_2"
assignvariableop_11_adam_decay*
&assignvariableop_12_adam_learning_rate 
assignvariableop_13_variable"
assignvariableop_14_variable_1
assignvariableop_15_total
assignvariableop_16_count
assignvariableop_17_total_1
assignvariableop_18_count_10
,assignvariableop_19_adam_conv2d_143_kernel_m.
*assignvariableop_20_adam_conv2d_143_bias_m/
+assignvariableop_21_adam_dense_308_kernel_m-
)assignvariableop_22_adam_dense_308_bias_m/
+assignvariableop_23_adam_dense_309_kernel_m-
)assignvariableop_24_adam_dense_309_bias_m/
+assignvariableop_25_adam_dense_310_kernel_m-
)assignvariableop_26_adam_dense_310_bias_m0
,assignvariableop_27_adam_conv2d_143_kernel_v.
*assignvariableop_28_adam_conv2d_143_bias_v/
+assignvariableop_29_adam_dense_308_kernel_v-
)assignvariableop_30_adam_dense_308_bias_v/
+assignvariableop_31_adam_dense_309_kernel_v-
)assignvariableop_32_adam_dense_309_bias_v/
+assignvariableop_33_adam_dense_310_kernel_v-
)assignvariableop_34_adam_dense_310_bias_v
identity_36??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*?
value?B?$B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB:layer-0/layer-0/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUEB:layer-0/layer-1/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*[
valueRBP$B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::*2
dtypes(
&2$			2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp"assignvariableop_conv2d_143_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp"assignvariableop_1_conv2d_143_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp#assignvariableop_2_dense_308_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp!assignvariableop_3_dense_308_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp#assignvariableop_4_dense_309_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp!assignvariableop_5_dense_309_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp#assignvariableop_6_dense_310_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp!assignvariableop_7_dense_310_biasIdentity_7:output:0"/device:CPU:0*
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
T0	*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_variableIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpassignvariableop_14_variable_1Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_totalIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpassignvariableop_16_countIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpassignvariableop_17_total_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOpassignvariableop_18_count_1Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp,assignvariableop_19_adam_conv2d_143_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp*assignvariableop_20_adam_conv2d_143_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp+assignvariableop_21_adam_dense_308_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp)assignvariableop_22_adam_dense_308_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp+assignvariableop_23_adam_dense_309_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp)assignvariableop_24_adam_dense_309_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp+assignvariableop_25_adam_dense_310_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp)assignvariableop_26_adam_dense_310_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp,assignvariableop_27_adam_conv2d_143_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp*assignvariableop_28_adam_conv2d_143_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp+assignvariableop_29_adam_dense_308_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp)assignvariableop_30_adam_dense_308_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp+assignvariableop_31_adam_dense_309_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp)assignvariableop_32_adam_dense_309_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp+assignvariableop_33_adam_dense_310_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp)assignvariableop_34_adam_dense_310_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_349
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_35Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_35?
Identity_36IdentityIdentity_35:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_36"#
identity_36Identity_36:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_34AssignVariableOp_342(
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
?
?
.__inference_Fourth_Model_layer_call_fn_2512477

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

*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_Fourth_Model_layer_call_and_return_conditional_losses_25121682
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
?
O
3__inference_max_pooling2d_132_layer_call_fn_2511883

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
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_max_pooling2d_132_layer_call_and_return_conditional_losses_25118772
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
??
?
W__inference_Training_Data_Augmentation_layer_call_and_return_conditional_losses_2511857

inputs=
9random_rotation_stateful_uniform_statefuluniform_resource
identity??0random_rotation/stateful_uniform/StatefulUniform?
5random_flip/random_flip_left_right/control_dependencyIdentityinputs*
T0*
_class
loc:@inputs*/
_output_shapes
:?????????27
5random_flip/random_flip_left_right/control_dependency?
(random_flip/random_flip_left_right/ShapeShape>random_flip/random_flip_left_right/control_dependency:output:0*
T0*
_output_shapes
:2*
(random_flip/random_flip_left_right/Shape?
6random_flip/random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 28
6random_flip/random_flip_left_right/strided_slice/stack?
8random_flip/random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8random_flip/random_flip_left_right/strided_slice/stack_1?
8random_flip/random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8random_flip/random_flip_left_right/strided_slice/stack_2?
0random_flip/random_flip_left_right/strided_sliceStridedSlice1random_flip/random_flip_left_right/Shape:output:0?random_flip/random_flip_left_right/strided_slice/stack:output:0Arandom_flip/random_flip_left_right/strided_slice/stack_1:output:0Arandom_flip/random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask22
0random_flip/random_flip_left_right/strided_slice?
7random_flip/random_flip_left_right/random_uniform/shapePack9random_flip/random_flip_left_right/strided_slice:output:0*
N*
T0*
_output_shapes
:29
7random_flip/random_flip_left_right/random_uniform/shape?
5random_flip/random_flip_left_right/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    27
5random_flip/random_flip_left_right/random_uniform/min?
5random_flip/random_flip_left_right/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ??27
5random_flip/random_flip_left_right/random_uniform/max?
?random_flip/random_flip_left_right/random_uniform/RandomUniformRandomUniform@random_flip/random_flip_left_right/random_uniform/shape:output:0*
T0*#
_output_shapes
:?????????*
dtype02A
?random_flip/random_flip_left_right/random_uniform/RandomUniform?
5random_flip/random_flip_left_right/random_uniform/MulMulHrandom_flip/random_flip_left_right/random_uniform/RandomUniform:output:0>random_flip/random_flip_left_right/random_uniform/max:output:0*
T0*#
_output_shapes
:?????????27
5random_flip/random_flip_left_right/random_uniform/Mul?
2random_flip/random_flip_left_right/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :24
2random_flip/random_flip_left_right/Reshape/shape/1?
2random_flip/random_flip_left_right/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :24
2random_flip/random_flip_left_right/Reshape/shape/2?
2random_flip/random_flip_left_right/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :24
2random_flip/random_flip_left_right/Reshape/shape/3?
0random_flip/random_flip_left_right/Reshape/shapePack9random_flip/random_flip_left_right/strided_slice:output:0;random_flip/random_flip_left_right/Reshape/shape/1:output:0;random_flip/random_flip_left_right/Reshape/shape/2:output:0;random_flip/random_flip_left_right/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:22
0random_flip/random_flip_left_right/Reshape/shape?
*random_flip/random_flip_left_right/ReshapeReshape9random_flip/random_flip_left_right/random_uniform/Mul:z:09random_flip/random_flip_left_right/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2,
*random_flip/random_flip_left_right/Reshape?
(random_flip/random_flip_left_right/RoundRound3random_flip/random_flip_left_right/Reshape:output:0*
T0*/
_output_shapes
:?????????2*
(random_flip/random_flip_left_right/Round?
1random_flip/random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:23
1random_flip/random_flip_left_right/ReverseV2/axis?
,random_flip/random_flip_left_right/ReverseV2	ReverseV2>random_flip/random_flip_left_right/control_dependency:output:0:random_flip/random_flip_left_right/ReverseV2/axis:output:0*
T0*/
_output_shapes
:?????????2.
,random_flip/random_flip_left_right/ReverseV2?
&random_flip/random_flip_left_right/mulMul,random_flip/random_flip_left_right/Round:y:05random_flip/random_flip_left_right/ReverseV2:output:0*
T0*/
_output_shapes
:?????????2(
&random_flip/random_flip_left_right/mul?
(random_flip/random_flip_left_right/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2*
(random_flip/random_flip_left_right/sub/x?
&random_flip/random_flip_left_right/subSub1random_flip/random_flip_left_right/sub/x:output:0,random_flip/random_flip_left_right/Round:y:0*
T0*/
_output_shapes
:?????????2(
&random_flip/random_flip_left_right/sub?
(random_flip/random_flip_left_right/mul_1Mul*random_flip/random_flip_left_right/sub:z:0>random_flip/random_flip_left_right/control_dependency:output:0*
T0*/
_output_shapes
:?????????2*
(random_flip/random_flip_left_right/mul_1?
&random_flip/random_flip_left_right/addAddV2*random_flip/random_flip_left_right/mul:z:0,random_flip/random_flip_left_right/mul_1:z:0*
T0*/
_output_shapes
:?????????2(
&random_flip/random_flip_left_right/add?
random_rotation/ShapeShape*random_flip/random_flip_left_right/add:z:0*
T0*
_output_shapes
:2
random_rotation/Shape?
#random_rotation/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#random_rotation/strided_slice/stack?
%random_rotation/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%random_rotation/strided_slice/stack_1?
%random_rotation/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%random_rotation/strided_slice/stack_2?
random_rotation/strided_sliceStridedSlicerandom_rotation/Shape:output:0,random_rotation/strided_slice/stack:output:0.random_rotation/strided_slice/stack_1:output:0.random_rotation/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
random_rotation/strided_slice?
%random_rotation/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2'
%random_rotation/strided_slice_1/stack?
'random_rotation/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'random_rotation/strided_slice_1/stack_1?
'random_rotation/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'random_rotation/strided_slice_1/stack_2?
random_rotation/strided_slice_1StridedSlicerandom_rotation/Shape:output:0.random_rotation/strided_slice_1/stack:output:00random_rotation/strided_slice_1/stack_1:output:00random_rotation/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
random_rotation/strided_slice_1?
random_rotation/CastCast(random_rotation/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
random_rotation/Cast?
%random_rotation/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2'
%random_rotation/strided_slice_2/stack?
'random_rotation/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'random_rotation/strided_slice_2/stack_1?
'random_rotation/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'random_rotation/strided_slice_2/stack_2?
random_rotation/strided_slice_2StridedSlicerandom_rotation/Shape:output:0.random_rotation/strided_slice_2/stack:output:00random_rotation/strided_slice_2/stack_1:output:00random_rotation/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
random_rotation/strided_slice_2?
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
&random_rotation/stateful_uniform/shape?
$random_rotation/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *|٠?2&
$random_rotation/stateful_uniform/min?
$random_rotation/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *|٠?2&
$random_rotation/stateful_uniform/max?
:random_rotation/stateful_uniform/StatefulUniform/algorithmConst*
_output_shapes
: *
dtype0	*
value	B	 R2<
:random_rotation/stateful_uniform/StatefulUniform/algorithm?
0random_rotation/stateful_uniform/StatefulUniformStatefulUniform9random_rotation_stateful_uniform_statefuluniform_resourceCrandom_rotation/stateful_uniform/StatefulUniform/algorithm:output:0/random_rotation/stateful_uniform/shape:output:0*#
_output_shapes
:?????????*
shape_dtype022
0random_rotation/stateful_uniform/StatefulUniform?
$random_rotation/stateful_uniform/subSub-random_rotation/stateful_uniform/max:output:0-random_rotation/stateful_uniform/min:output:0*
T0*
_output_shapes
: 2&
$random_rotation/stateful_uniform/sub?
$random_rotation/stateful_uniform/mulMul9random_rotation/stateful_uniform/StatefulUniform:output:0(random_rotation/stateful_uniform/sub:z:0*
T0*#
_output_shapes
:?????????2&
$random_rotation/stateful_uniform/mul?
 random_rotation/stateful_uniformAdd(random_rotation/stateful_uniform/mul:z:0-random_rotation/stateful_uniform/min:output:0*
T0*#
_output_shapes
:?????????2"
 random_rotation/stateful_uniform?
%random_rotation/rotation_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2'
%random_rotation/rotation_matrix/sub/y?
#random_rotation/rotation_matrix/subSubrandom_rotation/Cast_1:y:0.random_rotation/rotation_matrix/sub/y:output:0*
T0*
_output_shapes
: 2%
#random_rotation/rotation_matrix/sub?
#random_rotation/rotation_matrix/CosCos$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:?????????2%
#random_rotation/rotation_matrix/Cos?
'random_rotation/rotation_matrix/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2)
'random_rotation/rotation_matrix/sub_1/y?
%random_rotation/rotation_matrix/sub_1Subrandom_rotation/Cast_1:y:00random_rotation/rotation_matrix/sub_1/y:output:0*
T0*
_output_shapes
: 2'
%random_rotation/rotation_matrix/sub_1?
#random_rotation/rotation_matrix/mulMul'random_rotation/rotation_matrix/Cos:y:0)random_rotation/rotation_matrix/sub_1:z:0*
T0*#
_output_shapes
:?????????2%
#random_rotation/rotation_matrix/mul?
#random_rotation/rotation_matrix/SinSin$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:?????????2%
#random_rotation/rotation_matrix/Sin?
'random_rotation/rotation_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2)
'random_rotation/rotation_matrix/sub_2/y?
%random_rotation/rotation_matrix/sub_2Subrandom_rotation/Cast:y:00random_rotation/rotation_matrix/sub_2/y:output:0*
T0*
_output_shapes
: 2'
%random_rotation/rotation_matrix/sub_2?
%random_rotation/rotation_matrix/mul_1Mul'random_rotation/rotation_matrix/Sin:y:0)random_rotation/rotation_matrix/sub_2:z:0*
T0*#
_output_shapes
:?????????2'
%random_rotation/rotation_matrix/mul_1?
%random_rotation/rotation_matrix/sub_3Sub'random_rotation/rotation_matrix/mul:z:0)random_rotation/rotation_matrix/mul_1:z:0*
T0*#
_output_shapes
:?????????2'
%random_rotation/rotation_matrix/sub_3?
%random_rotation/rotation_matrix/sub_4Sub'random_rotation/rotation_matrix/sub:z:0)random_rotation/rotation_matrix/sub_3:z:0*
T0*#
_output_shapes
:?????????2'
%random_rotation/rotation_matrix/sub_4?
)random_rotation/rotation_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2+
)random_rotation/rotation_matrix/truediv/y?
'random_rotation/rotation_matrix/truedivRealDiv)random_rotation/rotation_matrix/sub_4:z:02random_rotation/rotation_matrix/truediv/y:output:0*
T0*#
_output_shapes
:?????????2)
'random_rotation/rotation_matrix/truediv?
'random_rotation/rotation_matrix/sub_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2)
'random_rotation/rotation_matrix/sub_5/y?
%random_rotation/rotation_matrix/sub_5Subrandom_rotation/Cast:y:00random_rotation/rotation_matrix/sub_5/y:output:0*
T0*
_output_shapes
: 2'
%random_rotation/rotation_matrix/sub_5?
%random_rotation/rotation_matrix/Sin_1Sin$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:?????????2'
%random_rotation/rotation_matrix/Sin_1?
'random_rotation/rotation_matrix/sub_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2)
'random_rotation/rotation_matrix/sub_6/y?
%random_rotation/rotation_matrix/sub_6Subrandom_rotation/Cast_1:y:00random_rotation/rotation_matrix/sub_6/y:output:0*
T0*
_output_shapes
: 2'
%random_rotation/rotation_matrix/sub_6?
%random_rotation/rotation_matrix/mul_2Mul)random_rotation/rotation_matrix/Sin_1:y:0)random_rotation/rotation_matrix/sub_6:z:0*
T0*#
_output_shapes
:?????????2'
%random_rotation/rotation_matrix/mul_2?
%random_rotation/rotation_matrix/Cos_1Cos$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:?????????2'
%random_rotation/rotation_matrix/Cos_1?
'random_rotation/rotation_matrix/sub_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2)
'random_rotation/rotation_matrix/sub_7/y?
%random_rotation/rotation_matrix/sub_7Subrandom_rotation/Cast:y:00random_rotation/rotation_matrix/sub_7/y:output:0*
T0*
_output_shapes
: 2'
%random_rotation/rotation_matrix/sub_7?
%random_rotation/rotation_matrix/mul_3Mul)random_rotation/rotation_matrix/Cos_1:y:0)random_rotation/rotation_matrix/sub_7:z:0*
T0*#
_output_shapes
:?????????2'
%random_rotation/rotation_matrix/mul_3?
#random_rotation/rotation_matrix/addAddV2)random_rotation/rotation_matrix/mul_2:z:0)random_rotation/rotation_matrix/mul_3:z:0*
T0*#
_output_shapes
:?????????2%
#random_rotation/rotation_matrix/add?
%random_rotation/rotation_matrix/sub_8Sub)random_rotation/rotation_matrix/sub_5:z:0'random_rotation/rotation_matrix/add:z:0*
T0*#
_output_shapes
:?????????2'
%random_rotation/rotation_matrix/sub_8?
+random_rotation/rotation_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2-
+random_rotation/rotation_matrix/truediv_1/y?
)random_rotation/rotation_matrix/truediv_1RealDiv)random_rotation/rotation_matrix/sub_8:z:04random_rotation/rotation_matrix/truediv_1/y:output:0*
T0*#
_output_shapes
:?????????2+
)random_rotation/rotation_matrix/truediv_1?
%random_rotation/rotation_matrix/ShapeShape$random_rotation/stateful_uniform:z:0*
T0*
_output_shapes
:2'
%random_rotation/rotation_matrix/Shape?
3random_rotation/rotation_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3random_rotation/rotation_matrix/strided_slice/stack?
5random_rotation/rotation_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5random_rotation/rotation_matrix/strided_slice/stack_1?
5random_rotation/rotation_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5random_rotation/rotation_matrix/strided_slice/stack_2?
-random_rotation/rotation_matrix/strided_sliceStridedSlice.random_rotation/rotation_matrix/Shape:output:0<random_rotation/rotation_matrix/strided_slice/stack:output:0>random_rotation/rotation_matrix/strided_slice/stack_1:output:0>random_rotation/rotation_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-random_rotation/rotation_matrix/strided_slice?
%random_rotation/rotation_matrix/Cos_2Cos$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:?????????2'
%random_rotation/rotation_matrix/Cos_2?
5random_rotation/rotation_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        27
5random_rotation/rotation_matrix/strided_slice_1/stack?
7random_rotation/rotation_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation/rotation_matrix/strided_slice_1/stack_1?
7random_rotation/rotation_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7random_rotation/rotation_matrix/strided_slice_1/stack_2?
/random_rotation/rotation_matrix/strided_slice_1StridedSlice)random_rotation/rotation_matrix/Cos_2:y:0>random_rotation/rotation_matrix/strided_slice_1/stack:output:0@random_rotation/rotation_matrix/strided_slice_1/stack_1:output:0@random_rotation/rotation_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
new_axis_mask21
/random_rotation/rotation_matrix/strided_slice_1?
%random_rotation/rotation_matrix/Sin_2Sin$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:?????????2'
%random_rotation/rotation_matrix/Sin_2?
5random_rotation/rotation_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        27
5random_rotation/rotation_matrix/strided_slice_2/stack?
7random_rotation/rotation_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation/rotation_matrix/strided_slice_2/stack_1?
7random_rotation/rotation_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7random_rotation/rotation_matrix/strided_slice_2/stack_2?
/random_rotation/rotation_matrix/strided_slice_2StridedSlice)random_rotation/rotation_matrix/Sin_2:y:0>random_rotation/rotation_matrix/strided_slice_2/stack:output:0@random_rotation/rotation_matrix/strided_slice_2/stack_1:output:0@random_rotation/rotation_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
new_axis_mask21
/random_rotation/rotation_matrix/strided_slice_2?
#random_rotation/rotation_matrix/NegNeg8random_rotation/rotation_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????2%
#random_rotation/rotation_matrix/Neg?
5random_rotation/rotation_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        27
5random_rotation/rotation_matrix/strided_slice_3/stack?
7random_rotation/rotation_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation/rotation_matrix/strided_slice_3/stack_1?
7random_rotation/rotation_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7random_rotation/rotation_matrix/strided_slice_3/stack_2?
/random_rotation/rotation_matrix/strided_slice_3StridedSlice+random_rotation/rotation_matrix/truediv:z:0>random_rotation/rotation_matrix/strided_slice_3/stack:output:0@random_rotation/rotation_matrix/strided_slice_3/stack_1:output:0@random_rotation/rotation_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
new_axis_mask21
/random_rotation/rotation_matrix/strided_slice_3?
%random_rotation/rotation_matrix/Sin_3Sin$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:?????????2'
%random_rotation/rotation_matrix/Sin_3?
5random_rotation/rotation_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        27
5random_rotation/rotation_matrix/strided_slice_4/stack?
7random_rotation/rotation_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation/rotation_matrix/strided_slice_4/stack_1?
7random_rotation/rotation_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7random_rotation/rotation_matrix/strided_slice_4/stack_2?
/random_rotation/rotation_matrix/strided_slice_4StridedSlice)random_rotation/rotation_matrix/Sin_3:y:0>random_rotation/rotation_matrix/strided_slice_4/stack:output:0@random_rotation/rotation_matrix/strided_slice_4/stack_1:output:0@random_rotation/rotation_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
new_axis_mask21
/random_rotation/rotation_matrix/strided_slice_4?
%random_rotation/rotation_matrix/Cos_3Cos$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:?????????2'
%random_rotation/rotation_matrix/Cos_3?
5random_rotation/rotation_matrix/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        27
5random_rotation/rotation_matrix/strided_slice_5/stack?
7random_rotation/rotation_matrix/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation/rotation_matrix/strided_slice_5/stack_1?
7random_rotation/rotation_matrix/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7random_rotation/rotation_matrix/strided_slice_5/stack_2?
/random_rotation/rotation_matrix/strided_slice_5StridedSlice)random_rotation/rotation_matrix/Cos_3:y:0>random_rotation/rotation_matrix/strided_slice_5/stack:output:0@random_rotation/rotation_matrix/strided_slice_5/stack_1:output:0@random_rotation/rotation_matrix/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
new_axis_mask21
/random_rotation/rotation_matrix/strided_slice_5?
5random_rotation/rotation_matrix/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        27
5random_rotation/rotation_matrix/strided_slice_6/stack?
7random_rotation/rotation_matrix/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation/rotation_matrix/strided_slice_6/stack_1?
7random_rotation/rotation_matrix/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7random_rotation/rotation_matrix/strided_slice_6/stack_2?
/random_rotation/rotation_matrix/strided_slice_6StridedSlice-random_rotation/rotation_matrix/truediv_1:z:0>random_rotation/rotation_matrix/strided_slice_6/stack:output:0@random_rotation/rotation_matrix/strided_slice_6/stack_1:output:0@random_rotation/rotation_matrix/strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
new_axis_mask21
/random_rotation/rotation_matrix/strided_slice_6?
+random_rotation/rotation_matrix/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2-
+random_rotation/rotation_matrix/zeros/mul/y?
)random_rotation/rotation_matrix/zeros/mulMul6random_rotation/rotation_matrix/strided_slice:output:04random_rotation/rotation_matrix/zeros/mul/y:output:0*
T0*
_output_shapes
: 2+
)random_rotation/rotation_matrix/zeros/mul?
,random_rotation/rotation_matrix/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2.
,random_rotation/rotation_matrix/zeros/Less/y?
*random_rotation/rotation_matrix/zeros/LessLess-random_rotation/rotation_matrix/zeros/mul:z:05random_rotation/rotation_matrix/zeros/Less/y:output:0*
T0*
_output_shapes
: 2,
*random_rotation/rotation_matrix/zeros/Less?
.random_rotation/rotation_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :20
.random_rotation/rotation_matrix/zeros/packed/1?
,random_rotation/rotation_matrix/zeros/packedPack6random_rotation/rotation_matrix/strided_slice:output:07random_rotation/rotation_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2.
,random_rotation/rotation_matrix/zeros/packed?
+random_rotation/rotation_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+random_rotation/rotation_matrix/zeros/Const?
%random_rotation/rotation_matrix/zerosFill5random_rotation/rotation_matrix/zeros/packed:output:04random_rotation/rotation_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:?????????2'
%random_rotation/rotation_matrix/zeros?
+random_rotation/rotation_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2-
+random_rotation/rotation_matrix/concat/axis?
&random_rotation/rotation_matrix/concatConcatV28random_rotation/rotation_matrix/strided_slice_1:output:0'random_rotation/rotation_matrix/Neg:y:08random_rotation/rotation_matrix/strided_slice_3:output:08random_rotation/rotation_matrix/strided_slice_4:output:08random_rotation/rotation_matrix/strided_slice_5:output:08random_rotation/rotation_matrix/strided_slice_6:output:0.random_rotation/rotation_matrix/zeros:output:04random_rotation/rotation_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2(
&random_rotation/rotation_matrix/concat?
random_rotation/transform/ShapeShape*random_flip/random_flip_left_right/add:z:0*
T0*
_output_shapes
:2!
random_rotation/transform/Shape?
-random_rotation/transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2/
-random_rotation/transform/strided_slice/stack?
/random_rotation/transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/random_rotation/transform/strided_slice/stack_1?
/random_rotation/transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/random_rotation/transform/strided_slice/stack_2?
'random_rotation/transform/strided_sliceStridedSlice(random_rotation/transform/Shape:output:06random_rotation/transform/strided_slice/stack:output:08random_rotation/transform/strided_slice/stack_1:output:08random_rotation/transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2)
'random_rotation/transform/strided_slice?
$random_rotation/transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2&
$random_rotation/transform/fill_value?
4random_rotation/transform/ImageProjectiveTransformV3ImageProjectiveTransformV3*random_flip/random_flip_left_right/add:z:0/random_rotation/rotation_matrix/concat:output:00random_rotation/transform/strided_slice:output:0-random_rotation/transform/fill_value:output:0*/
_output_shapes
:?????????*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR26
4random_rotation/transform/ImageProjectiveTransformV3?
$random_contrast/random_uniform/shapeConst*
_output_shapes
: *
dtype0*
valueB 2&
$random_contrast/random_uniform/shape?
"random_contrast/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2$
"random_contrast/random_uniform/min?
"random_contrast/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *?̌?2$
"random_contrast/random_uniform/max?
,random_contrast/random_uniform/RandomUniformRandomUniform-random_contrast/random_uniform/shape:output:0*
T0*
_output_shapes
: *
dtype02.
,random_contrast/random_uniform/RandomUniform?
"random_contrast/random_uniform/subSub+random_contrast/random_uniform/max:output:0+random_contrast/random_uniform/min:output:0*
T0*
_output_shapes
: 2$
"random_contrast/random_uniform/sub?
"random_contrast/random_uniform/mulMul5random_contrast/random_uniform/RandomUniform:output:0&random_contrast/random_uniform/sub:z:0*
T0*
_output_shapes
: 2$
"random_contrast/random_uniform/mul?
random_contrast/random_uniformAdd&random_contrast/random_uniform/mul:z:0+random_contrast/random_uniform/min:output:0*
T0*
_output_shapes
: 2 
random_contrast/random_uniform?
random_contrast/adjust_contrastAdjustContrastv2Irandom_rotation/transform/ImageProjectiveTransformV3:transformed_images:0"random_contrast/random_uniform:z:0*/
_output_shapes
:?????????2!
random_contrast/adjust_contrast?
(random_contrast/adjust_contrast/IdentityIdentity(random_contrast/adjust_contrast:output:0*
T0*/
_output_shapes
:?????????2*
(random_contrast/adjust_contrast/Identity?
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
??
?
I__inference_Fourth_Model_layer_call_and_return_conditional_losses_2512397

inputsX
Ttraining_data_augmentation_random_rotation_stateful_uniform_statefuluniform_resource-
)conv2d_143_conv2d_readvariableop_resource.
*conv2d_143_biasadd_readvariableop_resource,
(dense_308_matmul_readvariableop_resource-
)dense_308_biasadd_readvariableop_resource,
(dense_309_matmul_readvariableop_resource-
)dense_309_biasadd_readvariableop_resource,
(dense_310_matmul_readvariableop_resource-
)dense_310_biasadd_readvariableop_resource
identity??KTraining_Data_Augmentation/random_rotation/stateful_uniform/StatefulUniform?!conv2d_143/BiasAdd/ReadVariableOp? conv2d_143/Conv2D/ReadVariableOp? dense_308/BiasAdd/ReadVariableOp?dense_308/MatMul/ReadVariableOp? dense_309/BiasAdd/ReadVariableOp?dense_309/MatMul/ReadVariableOp? dense_310/BiasAdd/ReadVariableOp?dense_310/MatMul/ReadVariableOp?
PTraining_Data_Augmentation/random_flip/random_flip_left_right/control_dependencyIdentityinputs*
T0*
_class
loc:@inputs*/
_output_shapes
:?????????2R
PTraining_Data_Augmentation/random_flip/random_flip_left_right/control_dependency?
CTraining_Data_Augmentation/random_flip/random_flip_left_right/ShapeShapeYTraining_Data_Augmentation/random_flip/random_flip_left_right/control_dependency:output:0*
T0*
_output_shapes
:2E
CTraining_Data_Augmentation/random_flip/random_flip_left_right/Shape?
QTraining_Data_Augmentation/random_flip/random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2S
QTraining_Data_Augmentation/random_flip/random_flip_left_right/strided_slice/stack?
STraining_Data_Augmentation/random_flip/random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2U
STraining_Data_Augmentation/random_flip/random_flip_left_right/strided_slice/stack_1?
STraining_Data_Augmentation/random_flip/random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2U
STraining_Data_Augmentation/random_flip/random_flip_left_right/strided_slice/stack_2?
KTraining_Data_Augmentation/random_flip/random_flip_left_right/strided_sliceStridedSliceLTraining_Data_Augmentation/random_flip/random_flip_left_right/Shape:output:0ZTraining_Data_Augmentation/random_flip/random_flip_left_right/strided_slice/stack:output:0\Training_Data_Augmentation/random_flip/random_flip_left_right/strided_slice/stack_1:output:0\Training_Data_Augmentation/random_flip/random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2M
KTraining_Data_Augmentation/random_flip/random_flip_left_right/strided_slice?
RTraining_Data_Augmentation/random_flip/random_flip_left_right/random_uniform/shapePackTTraining_Data_Augmentation/random_flip/random_flip_left_right/strided_slice:output:0*
N*
T0*
_output_shapes
:2T
RTraining_Data_Augmentation/random_flip/random_flip_left_right/random_uniform/shape?
PTraining_Data_Augmentation/random_flip/random_flip_left_right/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2R
PTraining_Data_Augmentation/random_flip/random_flip_left_right/random_uniform/min?
PTraining_Data_Augmentation/random_flip/random_flip_left_right/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2R
PTraining_Data_Augmentation/random_flip/random_flip_left_right/random_uniform/max?
ZTraining_Data_Augmentation/random_flip/random_flip_left_right/random_uniform/RandomUniformRandomUniform[Training_Data_Augmentation/random_flip/random_flip_left_right/random_uniform/shape:output:0*
T0*#
_output_shapes
:?????????*
dtype02\
ZTraining_Data_Augmentation/random_flip/random_flip_left_right/random_uniform/RandomUniform?
PTraining_Data_Augmentation/random_flip/random_flip_left_right/random_uniform/MulMulcTraining_Data_Augmentation/random_flip/random_flip_left_right/random_uniform/RandomUniform:output:0YTraining_Data_Augmentation/random_flip/random_flip_left_right/random_uniform/max:output:0*
T0*#
_output_shapes
:?????????2R
PTraining_Data_Augmentation/random_flip/random_flip_left_right/random_uniform/Mul?
MTraining_Data_Augmentation/random_flip/random_flip_left_right/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2O
MTraining_Data_Augmentation/random_flip/random_flip_left_right/Reshape/shape/1?
MTraining_Data_Augmentation/random_flip/random_flip_left_right/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2O
MTraining_Data_Augmentation/random_flip/random_flip_left_right/Reshape/shape/2?
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
ETraining_Data_Augmentation/random_flip/random_flip_left_right/Reshape?
CTraining_Data_Augmentation/random_flip/random_flip_left_right/RoundRoundNTraining_Data_Augmentation/random_flip/random_flip_left_right/Reshape:output:0*
T0*/
_output_shapes
:?????????2E
CTraining_Data_Augmentation/random_flip/random_flip_left_right/Round?
LTraining_Data_Augmentation/random_flip/random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:2N
LTraining_Data_Augmentation/random_flip/random_flip_left_right/ReverseV2/axis?
GTraining_Data_Augmentation/random_flip/random_flip_left_right/ReverseV2	ReverseV2YTraining_Data_Augmentation/random_flip/random_flip_left_right/control_dependency:output:0UTraining_Data_Augmentation/random_flip/random_flip_left_right/ReverseV2/axis:output:0*
T0*/
_output_shapes
:?????????2I
GTraining_Data_Augmentation/random_flip/random_flip_left_right/ReverseV2?
ATraining_Data_Augmentation/random_flip/random_flip_left_right/mulMulGTraining_Data_Augmentation/random_flip/random_flip_left_right/Round:y:0PTraining_Data_Augmentation/random_flip/random_flip_left_right/ReverseV2:output:0*
T0*/
_output_shapes
:?????????2C
ATraining_Data_Augmentation/random_flip/random_flip_left_right/mul?
CTraining_Data_Augmentation/random_flip/random_flip_left_right/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2E
CTraining_Data_Augmentation/random_flip/random_flip_left_right/sub/x?
ATraining_Data_Augmentation/random_flip/random_flip_left_right/subSubLTraining_Data_Augmentation/random_flip/random_flip_left_right/sub/x:output:0GTraining_Data_Augmentation/random_flip/random_flip_left_right/Round:y:0*
T0*/
_output_shapes
:?????????2C
ATraining_Data_Augmentation/random_flip/random_flip_left_right/sub?
CTraining_Data_Augmentation/random_flip/random_flip_left_right/mul_1MulETraining_Data_Augmentation/random_flip/random_flip_left_right/sub:z:0YTraining_Data_Augmentation/random_flip/random_flip_left_right/control_dependency:output:0*
T0*/
_output_shapes
:?????????2E
CTraining_Data_Augmentation/random_flip/random_flip_left_right/mul_1?
ATraining_Data_Augmentation/random_flip/random_flip_left_right/addAddV2ETraining_Data_Augmentation/random_flip/random_flip_left_right/mul:z:0GTraining_Data_Augmentation/random_flip/random_flip_left_right/mul_1:z:0*
T0*/
_output_shapes
:?????????2C
ATraining_Data_Augmentation/random_flip/random_flip_left_right/add?
0Training_Data_Augmentation/random_rotation/ShapeShapeETraining_Data_Augmentation/random_flip/random_flip_left_right/add:z:0*
T0*
_output_shapes
:22
0Training_Data_Augmentation/random_rotation/Shape?
>Training_Data_Augmentation/random_rotation/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2@
>Training_Data_Augmentation/random_rotation/strided_slice/stack?
@Training_Data_Augmentation/random_rotation/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2B
@Training_Data_Augmentation/random_rotation/strided_slice/stack_1?
@Training_Data_Augmentation/random_rotation/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2B
@Training_Data_Augmentation/random_rotation/strided_slice/stack_2?
8Training_Data_Augmentation/random_rotation/strided_sliceStridedSlice9Training_Data_Augmentation/random_rotation/Shape:output:0GTraining_Data_Augmentation/random_rotation/strided_slice/stack:output:0ITraining_Data_Augmentation/random_rotation/strided_slice/stack_1:output:0ITraining_Data_Augmentation/random_rotation/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2:
8Training_Data_Augmentation/random_rotation/strided_slice?
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
BTraining_Data_Augmentation/random_rotation/strided_slice_1/stack_2?
:Training_Data_Augmentation/random_rotation/strided_slice_1StridedSlice9Training_Data_Augmentation/random_rotation/Shape:output:0ITraining_Data_Augmentation/random_rotation/strided_slice_1/stack:output:0KTraining_Data_Augmentation/random_rotation/strided_slice_1/stack_1:output:0KTraining_Data_Augmentation/random_rotation/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2<
:Training_Data_Augmentation/random_rotation/strided_slice_1?
/Training_Data_Augmentation/random_rotation/CastCastCTraining_Data_Augmentation/random_rotation/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 21
/Training_Data_Augmentation/random_rotation/Cast?
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
BTraining_Data_Augmentation/random_rotation/strided_slice_2/stack_2?
:Training_Data_Augmentation/random_rotation/strided_slice_2StridedSlice9Training_Data_Augmentation/random_rotation/Shape:output:0ITraining_Data_Augmentation/random_rotation/strided_slice_2/stack:output:0KTraining_Data_Augmentation/random_rotation/strided_slice_2/stack_1:output:0KTraining_Data_Augmentation/random_rotation/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2<
:Training_Data_Augmentation/random_rotation/strided_slice_2?
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
ATraining_Data_Augmentation/random_rotation/stateful_uniform/shape?
?Training_Data_Augmentation/random_rotation/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *|٠?2A
?Training_Data_Augmentation/random_rotation/stateful_uniform/min?
?Training_Data_Augmentation/random_rotation/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *|٠?2A
?Training_Data_Augmentation/random_rotation/stateful_uniform/max?
UTraining_Data_Augmentation/random_rotation/stateful_uniform/StatefulUniform/algorithmConst*
_output_shapes
: *
dtype0	*
value	B	 R2W
UTraining_Data_Augmentation/random_rotation/stateful_uniform/StatefulUniform/algorithm?
KTraining_Data_Augmentation/random_rotation/stateful_uniform/StatefulUniformStatefulUniformTtraining_data_augmentation_random_rotation_stateful_uniform_statefuluniform_resource^Training_Data_Augmentation/random_rotation/stateful_uniform/StatefulUniform/algorithm:output:0JTraining_Data_Augmentation/random_rotation/stateful_uniform/shape:output:0*#
_output_shapes
:?????????*
shape_dtype02M
KTraining_Data_Augmentation/random_rotation/stateful_uniform/StatefulUniform?
?Training_Data_Augmentation/random_rotation/stateful_uniform/subSubHTraining_Data_Augmentation/random_rotation/stateful_uniform/max:output:0HTraining_Data_Augmentation/random_rotation/stateful_uniform/min:output:0*
T0*
_output_shapes
: 2A
?Training_Data_Augmentation/random_rotation/stateful_uniform/sub?
?Training_Data_Augmentation/random_rotation/stateful_uniform/mulMulTTraining_Data_Augmentation/random_rotation/stateful_uniform/StatefulUniform:output:0CTraining_Data_Augmentation/random_rotation/stateful_uniform/sub:z:0*
T0*#
_output_shapes
:?????????2A
?Training_Data_Augmentation/random_rotation/stateful_uniform/mul?
;Training_Data_Augmentation/random_rotation/stateful_uniformAddCTraining_Data_Augmentation/random_rotation/stateful_uniform/mul:z:0HTraining_Data_Augmentation/random_rotation/stateful_uniform/min:output:0*
T0*#
_output_shapes
:?????????2=
;Training_Data_Augmentation/random_rotation/stateful_uniform?
@Training_Data_Augmentation/random_rotation/rotation_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2B
@Training_Data_Augmentation/random_rotation/rotation_matrix/sub/y?
>Training_Data_Augmentation/random_rotation/rotation_matrix/subSub5Training_Data_Augmentation/random_rotation/Cast_1:y:0ITraining_Data_Augmentation/random_rotation/rotation_matrix/sub/y:output:0*
T0*
_output_shapes
: 2@
>Training_Data_Augmentation/random_rotation/rotation_matrix/sub?
>Training_Data_Augmentation/random_rotation/rotation_matrix/CosCos?Training_Data_Augmentation/random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:?????????2@
>Training_Data_Augmentation/random_rotation/rotation_matrix/Cos?
BTraining_Data_Augmentation/random_rotation/rotation_matrix/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2D
BTraining_Data_Augmentation/random_rotation/rotation_matrix/sub_1/y?
@Training_Data_Augmentation/random_rotation/rotation_matrix/sub_1Sub5Training_Data_Augmentation/random_rotation/Cast_1:y:0KTraining_Data_Augmentation/random_rotation/rotation_matrix/sub_1/y:output:0*
T0*
_output_shapes
: 2B
@Training_Data_Augmentation/random_rotation/rotation_matrix/sub_1?
>Training_Data_Augmentation/random_rotation/rotation_matrix/mulMulBTraining_Data_Augmentation/random_rotation/rotation_matrix/Cos:y:0DTraining_Data_Augmentation/random_rotation/rotation_matrix/sub_1:z:0*
T0*#
_output_shapes
:?????????2@
>Training_Data_Augmentation/random_rotation/rotation_matrix/mul?
>Training_Data_Augmentation/random_rotation/rotation_matrix/SinSin?Training_Data_Augmentation/random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:?????????2@
>Training_Data_Augmentation/random_rotation/rotation_matrix/Sin?
BTraining_Data_Augmentation/random_rotation/rotation_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2D
BTraining_Data_Augmentation/random_rotation/rotation_matrix/sub_2/y?
@Training_Data_Augmentation/random_rotation/rotation_matrix/sub_2Sub3Training_Data_Augmentation/random_rotation/Cast:y:0KTraining_Data_Augmentation/random_rotation/rotation_matrix/sub_2/y:output:0*
T0*
_output_shapes
: 2B
@Training_Data_Augmentation/random_rotation/rotation_matrix/sub_2?
@Training_Data_Augmentation/random_rotation/rotation_matrix/mul_1MulBTraining_Data_Augmentation/random_rotation/rotation_matrix/Sin:y:0DTraining_Data_Augmentation/random_rotation/rotation_matrix/sub_2:z:0*
T0*#
_output_shapes
:?????????2B
@Training_Data_Augmentation/random_rotation/rotation_matrix/mul_1?
@Training_Data_Augmentation/random_rotation/rotation_matrix/sub_3SubBTraining_Data_Augmentation/random_rotation/rotation_matrix/mul:z:0DTraining_Data_Augmentation/random_rotation/rotation_matrix/mul_1:z:0*
T0*#
_output_shapes
:?????????2B
@Training_Data_Augmentation/random_rotation/rotation_matrix/sub_3?
@Training_Data_Augmentation/random_rotation/rotation_matrix/sub_4SubBTraining_Data_Augmentation/random_rotation/rotation_matrix/sub:z:0DTraining_Data_Augmentation/random_rotation/rotation_matrix/sub_3:z:0*
T0*#
_output_shapes
:?????????2B
@Training_Data_Augmentation/random_rotation/rotation_matrix/sub_4?
DTraining_Data_Augmentation/random_rotation/rotation_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2F
DTraining_Data_Augmentation/random_rotation/rotation_matrix/truediv/y?
BTraining_Data_Augmentation/random_rotation/rotation_matrix/truedivRealDivDTraining_Data_Augmentation/random_rotation/rotation_matrix/sub_4:z:0MTraining_Data_Augmentation/random_rotation/rotation_matrix/truediv/y:output:0*
T0*#
_output_shapes
:?????????2D
BTraining_Data_Augmentation/random_rotation/rotation_matrix/truediv?
BTraining_Data_Augmentation/random_rotation/rotation_matrix/sub_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2D
BTraining_Data_Augmentation/random_rotation/rotation_matrix/sub_5/y?
@Training_Data_Augmentation/random_rotation/rotation_matrix/sub_5Sub3Training_Data_Augmentation/random_rotation/Cast:y:0KTraining_Data_Augmentation/random_rotation/rotation_matrix/sub_5/y:output:0*
T0*
_output_shapes
: 2B
@Training_Data_Augmentation/random_rotation/rotation_matrix/sub_5?
@Training_Data_Augmentation/random_rotation/rotation_matrix/Sin_1Sin?Training_Data_Augmentation/random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:?????????2B
@Training_Data_Augmentation/random_rotation/rotation_matrix/Sin_1?
BTraining_Data_Augmentation/random_rotation/rotation_matrix/sub_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2D
BTraining_Data_Augmentation/random_rotation/rotation_matrix/sub_6/y?
@Training_Data_Augmentation/random_rotation/rotation_matrix/sub_6Sub5Training_Data_Augmentation/random_rotation/Cast_1:y:0KTraining_Data_Augmentation/random_rotation/rotation_matrix/sub_6/y:output:0*
T0*
_output_shapes
: 2B
@Training_Data_Augmentation/random_rotation/rotation_matrix/sub_6?
@Training_Data_Augmentation/random_rotation/rotation_matrix/mul_2MulDTraining_Data_Augmentation/random_rotation/rotation_matrix/Sin_1:y:0DTraining_Data_Augmentation/random_rotation/rotation_matrix/sub_6:z:0*
T0*#
_output_shapes
:?????????2B
@Training_Data_Augmentation/random_rotation/rotation_matrix/mul_2?
@Training_Data_Augmentation/random_rotation/rotation_matrix/Cos_1Cos?Training_Data_Augmentation/random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:?????????2B
@Training_Data_Augmentation/random_rotation/rotation_matrix/Cos_1?
BTraining_Data_Augmentation/random_rotation/rotation_matrix/sub_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2D
BTraining_Data_Augmentation/random_rotation/rotation_matrix/sub_7/y?
@Training_Data_Augmentation/random_rotation/rotation_matrix/sub_7Sub3Training_Data_Augmentation/random_rotation/Cast:y:0KTraining_Data_Augmentation/random_rotation/rotation_matrix/sub_7/y:output:0*
T0*
_output_shapes
: 2B
@Training_Data_Augmentation/random_rotation/rotation_matrix/sub_7?
@Training_Data_Augmentation/random_rotation/rotation_matrix/mul_3MulDTraining_Data_Augmentation/random_rotation/rotation_matrix/Cos_1:y:0DTraining_Data_Augmentation/random_rotation/rotation_matrix/sub_7:z:0*
T0*#
_output_shapes
:?????????2B
@Training_Data_Augmentation/random_rotation/rotation_matrix/mul_3?
>Training_Data_Augmentation/random_rotation/rotation_matrix/addAddV2DTraining_Data_Augmentation/random_rotation/rotation_matrix/mul_2:z:0DTraining_Data_Augmentation/random_rotation/rotation_matrix/mul_3:z:0*
T0*#
_output_shapes
:?????????2@
>Training_Data_Augmentation/random_rotation/rotation_matrix/add?
@Training_Data_Augmentation/random_rotation/rotation_matrix/sub_8SubDTraining_Data_Augmentation/random_rotation/rotation_matrix/sub_5:z:0BTraining_Data_Augmentation/random_rotation/rotation_matrix/add:z:0*
T0*#
_output_shapes
:?????????2B
@Training_Data_Augmentation/random_rotation/rotation_matrix/sub_8?
FTraining_Data_Augmentation/random_rotation/rotation_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2H
FTraining_Data_Augmentation/random_rotation/rotation_matrix/truediv_1/y?
DTraining_Data_Augmentation/random_rotation/rotation_matrix/truediv_1RealDivDTraining_Data_Augmentation/random_rotation/rotation_matrix/sub_8:z:0OTraining_Data_Augmentation/random_rotation/rotation_matrix/truediv_1/y:output:0*
T0*#
_output_shapes
:?????????2F
DTraining_Data_Augmentation/random_rotation/rotation_matrix/truediv_1?
@Training_Data_Augmentation/random_rotation/rotation_matrix/ShapeShape?Training_Data_Augmentation/random_rotation/stateful_uniform:z:0*
T0*
_output_shapes
:2B
@Training_Data_Augmentation/random_rotation/rotation_matrix/Shape?
NTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2P
NTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice/stack?
PTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2R
PTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice/stack_1?
PTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2R
PTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice/stack_2?
HTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_sliceStridedSliceITraining_Data_Augmentation/random_rotation/rotation_matrix/Shape:output:0WTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice/stack:output:0YTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice/stack_1:output:0YTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2J
HTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice?
@Training_Data_Augmentation/random_rotation/rotation_matrix/Cos_2Cos?Training_Data_Augmentation/random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:?????????2B
@Training_Data_Augmentation/random_rotation/rotation_matrix/Cos_2?
PTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2R
PTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_1/stack?
RTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2T
RTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_1/stack_1?
RTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2T
RTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_1/stack_2?
JTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_1StridedSliceDTraining_Data_Augmentation/random_rotation/rotation_matrix/Cos_2:y:0YTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_1/stack:output:0[Training_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_1/stack_1:output:0[Training_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
new_axis_mask2L
JTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_1?
@Training_Data_Augmentation/random_rotation/rotation_matrix/Sin_2Sin?Training_Data_Augmentation/random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:?????????2B
@Training_Data_Augmentation/random_rotation/rotation_matrix/Sin_2?
PTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2R
PTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_2/stack?
RTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2T
RTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_2/stack_1?
RTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2T
RTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_2/stack_2?
JTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_2StridedSliceDTraining_Data_Augmentation/random_rotation/rotation_matrix/Sin_2:y:0YTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_2/stack:output:0[Training_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_2/stack_1:output:0[Training_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
new_axis_mask2L
JTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_2?
>Training_Data_Augmentation/random_rotation/rotation_matrix/NegNegSTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????2@
>Training_Data_Augmentation/random_rotation/rotation_matrix/Neg?
PTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        2R
PTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_3/stack?
RTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2T
RTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_3/stack_1?
RTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2T
RTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_3/stack_2?
JTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_3StridedSliceFTraining_Data_Augmentation/random_rotation/rotation_matrix/truediv:z:0YTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_3/stack:output:0[Training_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_3/stack_1:output:0[Training_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
new_axis_mask2L
JTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_3?
@Training_Data_Augmentation/random_rotation/rotation_matrix/Sin_3Sin?Training_Data_Augmentation/random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:?????????2B
@Training_Data_Augmentation/random_rotation/rotation_matrix/Sin_3?
PTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        2R
PTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_4/stack?
RTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2T
RTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_4/stack_1?
RTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2T
RTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_4/stack_2?
JTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_4StridedSliceDTraining_Data_Augmentation/random_rotation/rotation_matrix/Sin_3:y:0YTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_4/stack:output:0[Training_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_4/stack_1:output:0[Training_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
new_axis_mask2L
JTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_4?
@Training_Data_Augmentation/random_rotation/rotation_matrix/Cos_3Cos?Training_Data_Augmentation/random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:?????????2B
@Training_Data_Augmentation/random_rotation/rotation_matrix/Cos_3?
PTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        2R
PTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_5/stack?
RTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2T
RTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_5/stack_1?
RTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2T
RTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_5/stack_2?
JTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_5StridedSliceDTraining_Data_Augmentation/random_rotation/rotation_matrix/Cos_3:y:0YTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_5/stack:output:0[Training_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_5/stack_1:output:0[Training_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
new_axis_mask2L
JTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_5?
PTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2R
PTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_6/stack?
RTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2T
RTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_6/stack_1?
RTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2T
RTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_6/stack_2?
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
FTraining_Data_Augmentation/random_rotation/rotation_matrix/zeros/mul/y?
DTraining_Data_Augmentation/random_rotation/rotation_matrix/zeros/mulMulQTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice:output:0OTraining_Data_Augmentation/random_rotation/rotation_matrix/zeros/mul/y:output:0*
T0*
_output_shapes
: 2F
DTraining_Data_Augmentation/random_rotation/rotation_matrix/zeros/mul?
GTraining_Data_Augmentation/random_rotation/rotation_matrix/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2I
GTraining_Data_Augmentation/random_rotation/rotation_matrix/zeros/Less/y?
ETraining_Data_Augmentation/random_rotation/rotation_matrix/zeros/LessLessHTraining_Data_Augmentation/random_rotation/rotation_matrix/zeros/mul:z:0PTraining_Data_Augmentation/random_rotation/rotation_matrix/zeros/Less/y:output:0*
T0*
_output_shapes
: 2G
ETraining_Data_Augmentation/random_rotation/rotation_matrix/zeros/Less?
ITraining_Data_Augmentation/random_rotation/rotation_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2K
ITraining_Data_Augmentation/random_rotation/rotation_matrix/zeros/packed/1?
GTraining_Data_Augmentation/random_rotation/rotation_matrix/zeros/packedPackQTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice:output:0RTraining_Data_Augmentation/random_rotation/rotation_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2I
GTraining_Data_Augmentation/random_rotation/rotation_matrix/zeros/packed?
FTraining_Data_Augmentation/random_rotation/rotation_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2H
FTraining_Data_Augmentation/random_rotation/rotation_matrix/zeros/Const?
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
FTraining_Data_Augmentation/random_rotation/rotation_matrix/concat/axis?
ATraining_Data_Augmentation/random_rotation/rotation_matrix/concatConcatV2STraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_1:output:0BTraining_Data_Augmentation/random_rotation/rotation_matrix/Neg:y:0STraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_3:output:0STraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_4:output:0STraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_5:output:0STraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_6:output:0ITraining_Data_Augmentation/random_rotation/rotation_matrix/zeros:output:0OTraining_Data_Augmentation/random_rotation/rotation_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2C
ATraining_Data_Augmentation/random_rotation/rotation_matrix/concat?
:Training_Data_Augmentation/random_rotation/transform/ShapeShapeETraining_Data_Augmentation/random_flip/random_flip_left_right/add:z:0*
T0*
_output_shapes
:2<
:Training_Data_Augmentation/random_rotation/transform/Shape?
HTraining_Data_Augmentation/random_rotation/transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2J
HTraining_Data_Augmentation/random_rotation/transform/strided_slice/stack?
JTraining_Data_Augmentation/random_rotation/transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2L
JTraining_Data_Augmentation/random_rotation/transform/strided_slice/stack_1?
JTraining_Data_Augmentation/random_rotation/transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2L
JTraining_Data_Augmentation/random_rotation/transform/strided_slice/stack_2?
BTraining_Data_Augmentation/random_rotation/transform/strided_sliceStridedSliceCTraining_Data_Augmentation/random_rotation/transform/Shape:output:0QTraining_Data_Augmentation/random_rotation/transform/strided_slice/stack:output:0STraining_Data_Augmentation/random_rotation/transform/strided_slice/stack_1:output:0STraining_Data_Augmentation/random_rotation/transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2D
BTraining_Data_Augmentation/random_rotation/transform/strided_slice?
?Training_Data_Augmentation/random_rotation/transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2A
?Training_Data_Augmentation/random_rotation/transform/fill_value?
OTraining_Data_Augmentation/random_rotation/transform/ImageProjectiveTransformV3ImageProjectiveTransformV3ETraining_Data_Augmentation/random_flip/random_flip_left_right/add:z:0JTraining_Data_Augmentation/random_rotation/rotation_matrix/concat:output:0KTraining_Data_Augmentation/random_rotation/transform/strided_slice:output:0HTraining_Data_Augmentation/random_rotation/transform/fill_value:output:0*/
_output_shapes
:?????????*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR2Q
OTraining_Data_Augmentation/random_rotation/transform/ImageProjectiveTransformV3?
?Training_Data_Augmentation/random_contrast/random_uniform/shapeConst*
_output_shapes
: *
dtype0*
valueB 2A
?Training_Data_Augmentation/random_contrast/random_uniform/shape?
=Training_Data_Augmentation/random_contrast/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2?
=Training_Data_Augmentation/random_contrast/random_uniform/min?
=Training_Data_Augmentation/random_contrast/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *?̌?2?
=Training_Data_Augmentation/random_contrast/random_uniform/max?
GTraining_Data_Augmentation/random_contrast/random_uniform/RandomUniformRandomUniformHTraining_Data_Augmentation/random_contrast/random_uniform/shape:output:0*
T0*
_output_shapes
: *
dtype02I
GTraining_Data_Augmentation/random_contrast/random_uniform/RandomUniform?
=Training_Data_Augmentation/random_contrast/random_uniform/subSubFTraining_Data_Augmentation/random_contrast/random_uniform/max:output:0FTraining_Data_Augmentation/random_contrast/random_uniform/min:output:0*
T0*
_output_shapes
: 2?
=Training_Data_Augmentation/random_contrast/random_uniform/sub?
=Training_Data_Augmentation/random_contrast/random_uniform/mulMulPTraining_Data_Augmentation/random_contrast/random_uniform/RandomUniform:output:0ATraining_Data_Augmentation/random_contrast/random_uniform/sub:z:0*
T0*
_output_shapes
: 2?
=Training_Data_Augmentation/random_contrast/random_uniform/mul?
9Training_Data_Augmentation/random_contrast/random_uniformAddATraining_Data_Augmentation/random_contrast/random_uniform/mul:z:0FTraining_Data_Augmentation/random_contrast/random_uniform/min:output:0*
T0*
_output_shapes
: 2;
9Training_Data_Augmentation/random_contrast/random_uniform?
:Training_Data_Augmentation/random_contrast/adjust_contrastAdjustContrastv2dTraining_Data_Augmentation/random_rotation/transform/ImageProjectiveTransformV3:transformed_images:0=Training_Data_Augmentation/random_contrast/random_uniform:z:0*/
_output_shapes
:?????????2<
:Training_Data_Augmentation/random_contrast/adjust_contrast?
CTraining_Data_Augmentation/random_contrast/adjust_contrast/IdentityIdentityCTraining_Data_Augmentation/random_contrast/adjust_contrast:output:0*
T0*/
_output_shapes
:?????????2E
CTraining_Data_Augmentation/random_contrast/adjust_contrast/Identity?
 conv2d_143/Conv2D/ReadVariableOpReadVariableOp)conv2d_143_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02"
 conv2d_143/Conv2D/ReadVariableOp?
conv2d_143/Conv2DConv2DLTraining_Data_Augmentation/random_contrast/adjust_contrast/Identity:output:0(conv2d_143/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
conv2d_143/Conv2D?
!conv2d_143/BiasAdd/ReadVariableOpReadVariableOp*conv2d_143_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02#
!conv2d_143/BiasAdd/ReadVariableOp?
conv2d_143/BiasAddBiasAddconv2d_143/Conv2D:output:0)conv2d_143/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_143/BiasAdd?
conv2d_143/ReluReluconv2d_143/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2d_143/Relu?
max_pooling2d_132/MaxPoolMaxPoolconv2d_143/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2
max_pooling2d_132/MaxPool{
dropout_110/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_110/dropout/Const?
dropout_110/dropout/MulMul"max_pooling2d_132/MaxPool:output:0"dropout_110/dropout/Const:output:0*
T0*/
_output_shapes
:????????? 2
dropout_110/dropout/Mul?
dropout_110/dropout/ShapeShape"max_pooling2d_132/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_110/dropout/Shape?
0dropout_110/dropout/random_uniform/RandomUniformRandomUniform"dropout_110/dropout/Shape:output:0*
T0*/
_output_shapes
:????????? *
dtype022
0dropout_110/dropout/random_uniform/RandomUniform?
"dropout_110/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2$
"dropout_110/dropout/GreaterEqual/y?
 dropout_110/dropout/GreaterEqualGreaterEqual9dropout_110/dropout/random_uniform/RandomUniform:output:0+dropout_110/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:????????? 2"
 dropout_110/dropout/GreaterEqual?
dropout_110/dropout/CastCast$dropout_110/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:????????? 2
dropout_110/dropout/Cast?
dropout_110/dropout/Mul_1Muldropout_110/dropout/Mul:z:0dropout_110/dropout/Cast:y:0*
T0*/
_output_shapes
:????????? 2
dropout_110/dropout/Mul_1u
flatten_99/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_99/Const?
flatten_99/ReshapeReshapedropout_110/dropout/Mul_1:z:0flatten_99/Const:output:0*
T0*(
_output_shapes
:??????????*2
flatten_99/Reshape?
dense_308/MatMul/ReadVariableOpReadVariableOp(dense_308_matmul_readvariableop_resource* 
_output_shapes
:
?*?*
dtype02!
dense_308/MatMul/ReadVariableOp?
dense_308/MatMulMatMulflatten_99/Reshape:output:0'dense_308/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_308/MatMul?
 dense_308/BiasAdd/ReadVariableOpReadVariableOp)dense_308_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 dense_308/BiasAdd/ReadVariableOp?
dense_308/BiasAddBiasAdddense_308/MatMul:product:0(dense_308/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_308/BiasAddw
dense_308/ReluReludense_308/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_308/Relu?
dense_309/MatMul/ReadVariableOpReadVariableOp(dense_309_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02!
dense_309/MatMul/ReadVariableOp?
dense_309/MatMulMatMuldense_308/Relu:activations:0'dense_309/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_309/MatMul?
 dense_309/BiasAdd/ReadVariableOpReadVariableOp)dense_309_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 dense_309/BiasAdd/ReadVariableOp?
dense_309/BiasAddBiasAdddense_309/MatMul:product:0(dense_309/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_309/BiasAddw
dense_309/ReluReludense_309/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_309/Relu?
dense_310/MatMul/ReadVariableOpReadVariableOp(dense_310_matmul_readvariableop_resource*
_output_shapes
:	?
*
dtype02!
dense_310/MatMul/ReadVariableOp?
dense_310/MatMulMatMuldense_309/Relu:activations:0'dense_310/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_310/MatMul?
 dense_310/BiasAdd/ReadVariableOpReadVariableOp)dense_310_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02"
 dense_310/BiasAdd/ReadVariableOp?
dense_310/BiasAddBiasAdddense_310/MatMul:product:0(dense_310/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_310/BiasAdd
dense_310/SoftmaxSoftmaxdense_310/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
dense_310/Softmax?
IdentityIdentitydense_310/Softmax:softmax:0L^Training_Data_Augmentation/random_rotation/stateful_uniform/StatefulUniform"^conv2d_143/BiasAdd/ReadVariableOp!^conv2d_143/Conv2D/ReadVariableOp!^dense_308/BiasAdd/ReadVariableOp ^dense_308/MatMul/ReadVariableOp!^dense_309/BiasAdd/ReadVariableOp ^dense_309/MatMul/ReadVariableOp!^dense_310/BiasAdd/ReadVariableOp ^dense_310/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*R
_input_shapesA
?:?????????:::::::::2?
KTraining_Data_Augmentation/random_rotation/stateful_uniform/StatefulUniformKTraining_Data_Augmentation/random_rotation/stateful_uniform/StatefulUniform2F
!conv2d_143/BiasAdd/ReadVariableOp!conv2d_143/BiasAdd/ReadVariableOp2D
 conv2d_143/Conv2D/ReadVariableOp conv2d_143/Conv2D/ReadVariableOp2D
 dense_308/BiasAdd/ReadVariableOp dense_308/BiasAdd/ReadVariableOp2B
dense_308/MatMul/ReadVariableOpdense_308/MatMul/ReadVariableOp2D
 dense_309/BiasAdd/ReadVariableOp dense_309/BiasAdd/ReadVariableOp2B
dense_309/MatMul/ReadVariableOpdense_309/MatMul/ReadVariableOp2D
 dense_310/BiasAdd/ReadVariableOp dense_310/BiasAdd/ReadVariableOp2B
dense_310/MatMul/ReadVariableOpdense_310/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?-
?
I__inference_Fourth_Model_layer_call_and_return_conditional_losses_2512433

inputs-
)conv2d_143_conv2d_readvariableop_resource.
*conv2d_143_biasadd_readvariableop_resource,
(dense_308_matmul_readvariableop_resource-
)dense_308_biasadd_readvariableop_resource,
(dense_309_matmul_readvariableop_resource-
)dense_309_biasadd_readvariableop_resource,
(dense_310_matmul_readvariableop_resource-
)dense_310_biasadd_readvariableop_resource
identity??!conv2d_143/BiasAdd/ReadVariableOp? conv2d_143/Conv2D/ReadVariableOp? dense_308/BiasAdd/ReadVariableOp?dense_308/MatMul/ReadVariableOp? dense_309/BiasAdd/ReadVariableOp?dense_309/MatMul/ReadVariableOp? dense_310/BiasAdd/ReadVariableOp?dense_310/MatMul/ReadVariableOp?
 conv2d_143/Conv2D/ReadVariableOpReadVariableOp)conv2d_143_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02"
 conv2d_143/Conv2D/ReadVariableOp?
conv2d_143/Conv2DConv2Dinputs(conv2d_143/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
conv2d_143/Conv2D?
!conv2d_143/BiasAdd/ReadVariableOpReadVariableOp*conv2d_143_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02#
!conv2d_143/BiasAdd/ReadVariableOp?
conv2d_143/BiasAddBiasAddconv2d_143/Conv2D:output:0)conv2d_143/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_143/BiasAdd?
conv2d_143/ReluReluconv2d_143/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2d_143/Relu?
max_pooling2d_132/MaxPoolMaxPoolconv2d_143/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2
max_pooling2d_132/MaxPool?
dropout_110/IdentityIdentity"max_pooling2d_132/MaxPool:output:0*
T0*/
_output_shapes
:????????? 2
dropout_110/Identityu
flatten_99/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_99/Const?
flatten_99/ReshapeReshapedropout_110/Identity:output:0flatten_99/Const:output:0*
T0*(
_output_shapes
:??????????*2
flatten_99/Reshape?
dense_308/MatMul/ReadVariableOpReadVariableOp(dense_308_matmul_readvariableop_resource* 
_output_shapes
:
?*?*
dtype02!
dense_308/MatMul/ReadVariableOp?
dense_308/MatMulMatMulflatten_99/Reshape:output:0'dense_308/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_308/MatMul?
 dense_308/BiasAdd/ReadVariableOpReadVariableOp)dense_308_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 dense_308/BiasAdd/ReadVariableOp?
dense_308/BiasAddBiasAdddense_308/MatMul:product:0(dense_308/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_308/BiasAddw
dense_308/ReluReludense_308/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_308/Relu?
dense_309/MatMul/ReadVariableOpReadVariableOp(dense_309_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02!
dense_309/MatMul/ReadVariableOp?
dense_309/MatMulMatMuldense_308/Relu:activations:0'dense_309/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_309/MatMul?
 dense_309/BiasAdd/ReadVariableOpReadVariableOp)dense_309_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 dense_309/BiasAdd/ReadVariableOp?
dense_309/BiasAddBiasAdddense_309/MatMul:product:0(dense_309/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_309/BiasAddw
dense_309/ReluReludense_309/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_309/Relu?
dense_310/MatMul/ReadVariableOpReadVariableOp(dense_310_matmul_readvariableop_resource*
_output_shapes
:	?
*
dtype02!
dense_310/MatMul/ReadVariableOp?
dense_310/MatMulMatMuldense_309/Relu:activations:0'dense_310/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_310/MatMul?
 dense_310/BiasAdd/ReadVariableOpReadVariableOp)dense_310_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02"
 dense_310/BiasAdd/ReadVariableOp?
dense_310/BiasAddBiasAdddense_310/MatMul:product:0(dense_310/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_310/BiasAdd
dense_310/SoftmaxSoftmaxdense_310/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
dense_310/Softmax?
IdentityIdentitydense_310/Softmax:softmax:0"^conv2d_143/BiasAdd/ReadVariableOp!^conv2d_143/Conv2D/ReadVariableOp!^dense_308/BiasAdd/ReadVariableOp ^dense_308/MatMul/ReadVariableOp!^dense_309/BiasAdd/ReadVariableOp ^dense_309/MatMul/ReadVariableOp!^dense_310/BiasAdd/ReadVariableOp ^dense_310/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::2F
!conv2d_143/BiasAdd/ReadVariableOp!conv2d_143/BiasAdd/ReadVariableOp2D
 conv2d_143/Conv2D/ReadVariableOp conv2d_143/Conv2D/ReadVariableOp2D
 dense_308/BiasAdd/ReadVariableOp dense_308/BiasAdd/ReadVariableOp2B
dense_308/MatMul/ReadVariableOpdense_308/MatMul/ReadVariableOp2D
 dense_309/BiasAdd/ReadVariableOp dense_309/BiasAdd/ReadVariableOp2B
dense_309/MatMul/ReadVariableOpdense_309/MatMul/ReadVariableOp2D
 dense_310/BiasAdd/ReadVariableOp dense_310/BiasAdd/ReadVariableOp2B
dense_310/MatMul/ReadVariableOpdense_310/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
f
H__inference_dropout_110_layer_call_and_return_conditional_losses_2512670

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
?
c
<__inference_Training_Data_Augmentation_layer_call_fn_2511871
random_flip_input
identity?
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
GPU 2J 8? *`
f[RY
W__inference_Training_Data_Augmentation_layer_call_and_return_conditional_losses_25118682
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
?
s
W__inference_Training_Data_Augmentation_layer_call_and_return_conditional_losses_2512621

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
?
s
W__inference_Training_Data_Augmentation_layer_call_and_return_conditional_losses_2511868

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
??
?
W__inference_Training_Data_Augmentation_layer_call_and_return_conditional_losses_2511710
random_flip_input=
9random_rotation_stateful_uniform_statefuluniform_resource
identity??0random_rotation/stateful_uniform/StatefulUniform?
5random_flip/random_flip_left_right/control_dependencyIdentityrandom_flip_input*
T0*$
_class
loc:@random_flip_input*/
_output_shapes
:?????????27
5random_flip/random_flip_left_right/control_dependency?
(random_flip/random_flip_left_right/ShapeShape>random_flip/random_flip_left_right/control_dependency:output:0*
T0*
_output_shapes
:2*
(random_flip/random_flip_left_right/Shape?
6random_flip/random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 28
6random_flip/random_flip_left_right/strided_slice/stack?
8random_flip/random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8random_flip/random_flip_left_right/strided_slice/stack_1?
8random_flip/random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8random_flip/random_flip_left_right/strided_slice/stack_2?
0random_flip/random_flip_left_right/strided_sliceStridedSlice1random_flip/random_flip_left_right/Shape:output:0?random_flip/random_flip_left_right/strided_slice/stack:output:0Arandom_flip/random_flip_left_right/strided_slice/stack_1:output:0Arandom_flip/random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask22
0random_flip/random_flip_left_right/strided_slice?
7random_flip/random_flip_left_right/random_uniform/shapePack9random_flip/random_flip_left_right/strided_slice:output:0*
N*
T0*
_output_shapes
:29
7random_flip/random_flip_left_right/random_uniform/shape?
5random_flip/random_flip_left_right/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    27
5random_flip/random_flip_left_right/random_uniform/min?
5random_flip/random_flip_left_right/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ??27
5random_flip/random_flip_left_right/random_uniform/max?
?random_flip/random_flip_left_right/random_uniform/RandomUniformRandomUniform@random_flip/random_flip_left_right/random_uniform/shape:output:0*
T0*#
_output_shapes
:?????????*
dtype02A
?random_flip/random_flip_left_right/random_uniform/RandomUniform?
5random_flip/random_flip_left_right/random_uniform/MulMulHrandom_flip/random_flip_left_right/random_uniform/RandomUniform:output:0>random_flip/random_flip_left_right/random_uniform/max:output:0*
T0*#
_output_shapes
:?????????27
5random_flip/random_flip_left_right/random_uniform/Mul?
2random_flip/random_flip_left_right/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :24
2random_flip/random_flip_left_right/Reshape/shape/1?
2random_flip/random_flip_left_right/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :24
2random_flip/random_flip_left_right/Reshape/shape/2?
2random_flip/random_flip_left_right/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :24
2random_flip/random_flip_left_right/Reshape/shape/3?
0random_flip/random_flip_left_right/Reshape/shapePack9random_flip/random_flip_left_right/strided_slice:output:0;random_flip/random_flip_left_right/Reshape/shape/1:output:0;random_flip/random_flip_left_right/Reshape/shape/2:output:0;random_flip/random_flip_left_right/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:22
0random_flip/random_flip_left_right/Reshape/shape?
*random_flip/random_flip_left_right/ReshapeReshape9random_flip/random_flip_left_right/random_uniform/Mul:z:09random_flip/random_flip_left_right/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2,
*random_flip/random_flip_left_right/Reshape?
(random_flip/random_flip_left_right/RoundRound3random_flip/random_flip_left_right/Reshape:output:0*
T0*/
_output_shapes
:?????????2*
(random_flip/random_flip_left_right/Round?
1random_flip/random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:23
1random_flip/random_flip_left_right/ReverseV2/axis?
,random_flip/random_flip_left_right/ReverseV2	ReverseV2>random_flip/random_flip_left_right/control_dependency:output:0:random_flip/random_flip_left_right/ReverseV2/axis:output:0*
T0*/
_output_shapes
:?????????2.
,random_flip/random_flip_left_right/ReverseV2?
&random_flip/random_flip_left_right/mulMul,random_flip/random_flip_left_right/Round:y:05random_flip/random_flip_left_right/ReverseV2:output:0*
T0*/
_output_shapes
:?????????2(
&random_flip/random_flip_left_right/mul?
(random_flip/random_flip_left_right/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2*
(random_flip/random_flip_left_right/sub/x?
&random_flip/random_flip_left_right/subSub1random_flip/random_flip_left_right/sub/x:output:0,random_flip/random_flip_left_right/Round:y:0*
T0*/
_output_shapes
:?????????2(
&random_flip/random_flip_left_right/sub?
(random_flip/random_flip_left_right/mul_1Mul*random_flip/random_flip_left_right/sub:z:0>random_flip/random_flip_left_right/control_dependency:output:0*
T0*/
_output_shapes
:?????????2*
(random_flip/random_flip_left_right/mul_1?
&random_flip/random_flip_left_right/addAddV2*random_flip/random_flip_left_right/mul:z:0,random_flip/random_flip_left_right/mul_1:z:0*
T0*/
_output_shapes
:?????????2(
&random_flip/random_flip_left_right/add?
random_rotation/ShapeShape*random_flip/random_flip_left_right/add:z:0*
T0*
_output_shapes
:2
random_rotation/Shape?
#random_rotation/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#random_rotation/strided_slice/stack?
%random_rotation/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%random_rotation/strided_slice/stack_1?
%random_rotation/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%random_rotation/strided_slice/stack_2?
random_rotation/strided_sliceStridedSlicerandom_rotation/Shape:output:0,random_rotation/strided_slice/stack:output:0.random_rotation/strided_slice/stack_1:output:0.random_rotation/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
random_rotation/strided_slice?
%random_rotation/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2'
%random_rotation/strided_slice_1/stack?
'random_rotation/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'random_rotation/strided_slice_1/stack_1?
'random_rotation/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'random_rotation/strided_slice_1/stack_2?
random_rotation/strided_slice_1StridedSlicerandom_rotation/Shape:output:0.random_rotation/strided_slice_1/stack:output:00random_rotation/strided_slice_1/stack_1:output:00random_rotation/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
random_rotation/strided_slice_1?
random_rotation/CastCast(random_rotation/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
random_rotation/Cast?
%random_rotation/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2'
%random_rotation/strided_slice_2/stack?
'random_rotation/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'random_rotation/strided_slice_2/stack_1?
'random_rotation/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'random_rotation/strided_slice_2/stack_2?
random_rotation/strided_slice_2StridedSlicerandom_rotation/Shape:output:0.random_rotation/strided_slice_2/stack:output:00random_rotation/strided_slice_2/stack_1:output:00random_rotation/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
random_rotation/strided_slice_2?
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
&random_rotation/stateful_uniform/shape?
$random_rotation/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *|٠?2&
$random_rotation/stateful_uniform/min?
$random_rotation/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *|٠?2&
$random_rotation/stateful_uniform/max?
:random_rotation/stateful_uniform/StatefulUniform/algorithmConst*
_output_shapes
: *
dtype0	*
value	B	 R2<
:random_rotation/stateful_uniform/StatefulUniform/algorithm?
0random_rotation/stateful_uniform/StatefulUniformStatefulUniform9random_rotation_stateful_uniform_statefuluniform_resourceCrandom_rotation/stateful_uniform/StatefulUniform/algorithm:output:0/random_rotation/stateful_uniform/shape:output:0*#
_output_shapes
:?????????*
shape_dtype022
0random_rotation/stateful_uniform/StatefulUniform?
$random_rotation/stateful_uniform/subSub-random_rotation/stateful_uniform/max:output:0-random_rotation/stateful_uniform/min:output:0*
T0*
_output_shapes
: 2&
$random_rotation/stateful_uniform/sub?
$random_rotation/stateful_uniform/mulMul9random_rotation/stateful_uniform/StatefulUniform:output:0(random_rotation/stateful_uniform/sub:z:0*
T0*#
_output_shapes
:?????????2&
$random_rotation/stateful_uniform/mul?
 random_rotation/stateful_uniformAdd(random_rotation/stateful_uniform/mul:z:0-random_rotation/stateful_uniform/min:output:0*
T0*#
_output_shapes
:?????????2"
 random_rotation/stateful_uniform?
%random_rotation/rotation_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2'
%random_rotation/rotation_matrix/sub/y?
#random_rotation/rotation_matrix/subSubrandom_rotation/Cast_1:y:0.random_rotation/rotation_matrix/sub/y:output:0*
T0*
_output_shapes
: 2%
#random_rotation/rotation_matrix/sub?
#random_rotation/rotation_matrix/CosCos$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:?????????2%
#random_rotation/rotation_matrix/Cos?
'random_rotation/rotation_matrix/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2)
'random_rotation/rotation_matrix/sub_1/y?
%random_rotation/rotation_matrix/sub_1Subrandom_rotation/Cast_1:y:00random_rotation/rotation_matrix/sub_1/y:output:0*
T0*
_output_shapes
: 2'
%random_rotation/rotation_matrix/sub_1?
#random_rotation/rotation_matrix/mulMul'random_rotation/rotation_matrix/Cos:y:0)random_rotation/rotation_matrix/sub_1:z:0*
T0*#
_output_shapes
:?????????2%
#random_rotation/rotation_matrix/mul?
#random_rotation/rotation_matrix/SinSin$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:?????????2%
#random_rotation/rotation_matrix/Sin?
'random_rotation/rotation_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2)
'random_rotation/rotation_matrix/sub_2/y?
%random_rotation/rotation_matrix/sub_2Subrandom_rotation/Cast:y:00random_rotation/rotation_matrix/sub_2/y:output:0*
T0*
_output_shapes
: 2'
%random_rotation/rotation_matrix/sub_2?
%random_rotation/rotation_matrix/mul_1Mul'random_rotation/rotation_matrix/Sin:y:0)random_rotation/rotation_matrix/sub_2:z:0*
T0*#
_output_shapes
:?????????2'
%random_rotation/rotation_matrix/mul_1?
%random_rotation/rotation_matrix/sub_3Sub'random_rotation/rotation_matrix/mul:z:0)random_rotation/rotation_matrix/mul_1:z:0*
T0*#
_output_shapes
:?????????2'
%random_rotation/rotation_matrix/sub_3?
%random_rotation/rotation_matrix/sub_4Sub'random_rotation/rotation_matrix/sub:z:0)random_rotation/rotation_matrix/sub_3:z:0*
T0*#
_output_shapes
:?????????2'
%random_rotation/rotation_matrix/sub_4?
)random_rotation/rotation_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2+
)random_rotation/rotation_matrix/truediv/y?
'random_rotation/rotation_matrix/truedivRealDiv)random_rotation/rotation_matrix/sub_4:z:02random_rotation/rotation_matrix/truediv/y:output:0*
T0*#
_output_shapes
:?????????2)
'random_rotation/rotation_matrix/truediv?
'random_rotation/rotation_matrix/sub_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2)
'random_rotation/rotation_matrix/sub_5/y?
%random_rotation/rotation_matrix/sub_5Subrandom_rotation/Cast:y:00random_rotation/rotation_matrix/sub_5/y:output:0*
T0*
_output_shapes
: 2'
%random_rotation/rotation_matrix/sub_5?
%random_rotation/rotation_matrix/Sin_1Sin$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:?????????2'
%random_rotation/rotation_matrix/Sin_1?
'random_rotation/rotation_matrix/sub_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2)
'random_rotation/rotation_matrix/sub_6/y?
%random_rotation/rotation_matrix/sub_6Subrandom_rotation/Cast_1:y:00random_rotation/rotation_matrix/sub_6/y:output:0*
T0*
_output_shapes
: 2'
%random_rotation/rotation_matrix/sub_6?
%random_rotation/rotation_matrix/mul_2Mul)random_rotation/rotation_matrix/Sin_1:y:0)random_rotation/rotation_matrix/sub_6:z:0*
T0*#
_output_shapes
:?????????2'
%random_rotation/rotation_matrix/mul_2?
%random_rotation/rotation_matrix/Cos_1Cos$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:?????????2'
%random_rotation/rotation_matrix/Cos_1?
'random_rotation/rotation_matrix/sub_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2)
'random_rotation/rotation_matrix/sub_7/y?
%random_rotation/rotation_matrix/sub_7Subrandom_rotation/Cast:y:00random_rotation/rotation_matrix/sub_7/y:output:0*
T0*
_output_shapes
: 2'
%random_rotation/rotation_matrix/sub_7?
%random_rotation/rotation_matrix/mul_3Mul)random_rotation/rotation_matrix/Cos_1:y:0)random_rotation/rotation_matrix/sub_7:z:0*
T0*#
_output_shapes
:?????????2'
%random_rotation/rotation_matrix/mul_3?
#random_rotation/rotation_matrix/addAddV2)random_rotation/rotation_matrix/mul_2:z:0)random_rotation/rotation_matrix/mul_3:z:0*
T0*#
_output_shapes
:?????????2%
#random_rotation/rotation_matrix/add?
%random_rotation/rotation_matrix/sub_8Sub)random_rotation/rotation_matrix/sub_5:z:0'random_rotation/rotation_matrix/add:z:0*
T0*#
_output_shapes
:?????????2'
%random_rotation/rotation_matrix/sub_8?
+random_rotation/rotation_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2-
+random_rotation/rotation_matrix/truediv_1/y?
)random_rotation/rotation_matrix/truediv_1RealDiv)random_rotation/rotation_matrix/sub_8:z:04random_rotation/rotation_matrix/truediv_1/y:output:0*
T0*#
_output_shapes
:?????????2+
)random_rotation/rotation_matrix/truediv_1?
%random_rotation/rotation_matrix/ShapeShape$random_rotation/stateful_uniform:z:0*
T0*
_output_shapes
:2'
%random_rotation/rotation_matrix/Shape?
3random_rotation/rotation_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3random_rotation/rotation_matrix/strided_slice/stack?
5random_rotation/rotation_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5random_rotation/rotation_matrix/strided_slice/stack_1?
5random_rotation/rotation_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5random_rotation/rotation_matrix/strided_slice/stack_2?
-random_rotation/rotation_matrix/strided_sliceStridedSlice.random_rotation/rotation_matrix/Shape:output:0<random_rotation/rotation_matrix/strided_slice/stack:output:0>random_rotation/rotation_matrix/strided_slice/stack_1:output:0>random_rotation/rotation_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-random_rotation/rotation_matrix/strided_slice?
%random_rotation/rotation_matrix/Cos_2Cos$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:?????????2'
%random_rotation/rotation_matrix/Cos_2?
5random_rotation/rotation_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        27
5random_rotation/rotation_matrix/strided_slice_1/stack?
7random_rotation/rotation_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation/rotation_matrix/strided_slice_1/stack_1?
7random_rotation/rotation_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7random_rotation/rotation_matrix/strided_slice_1/stack_2?
/random_rotation/rotation_matrix/strided_slice_1StridedSlice)random_rotation/rotation_matrix/Cos_2:y:0>random_rotation/rotation_matrix/strided_slice_1/stack:output:0@random_rotation/rotation_matrix/strided_slice_1/stack_1:output:0@random_rotation/rotation_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
new_axis_mask21
/random_rotation/rotation_matrix/strided_slice_1?
%random_rotation/rotation_matrix/Sin_2Sin$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:?????????2'
%random_rotation/rotation_matrix/Sin_2?
5random_rotation/rotation_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        27
5random_rotation/rotation_matrix/strided_slice_2/stack?
7random_rotation/rotation_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation/rotation_matrix/strided_slice_2/stack_1?
7random_rotation/rotation_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7random_rotation/rotation_matrix/strided_slice_2/stack_2?
/random_rotation/rotation_matrix/strided_slice_2StridedSlice)random_rotation/rotation_matrix/Sin_2:y:0>random_rotation/rotation_matrix/strided_slice_2/stack:output:0@random_rotation/rotation_matrix/strided_slice_2/stack_1:output:0@random_rotation/rotation_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
new_axis_mask21
/random_rotation/rotation_matrix/strided_slice_2?
#random_rotation/rotation_matrix/NegNeg8random_rotation/rotation_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????2%
#random_rotation/rotation_matrix/Neg?
5random_rotation/rotation_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        27
5random_rotation/rotation_matrix/strided_slice_3/stack?
7random_rotation/rotation_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation/rotation_matrix/strided_slice_3/stack_1?
7random_rotation/rotation_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7random_rotation/rotation_matrix/strided_slice_3/stack_2?
/random_rotation/rotation_matrix/strided_slice_3StridedSlice+random_rotation/rotation_matrix/truediv:z:0>random_rotation/rotation_matrix/strided_slice_3/stack:output:0@random_rotation/rotation_matrix/strided_slice_3/stack_1:output:0@random_rotation/rotation_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
new_axis_mask21
/random_rotation/rotation_matrix/strided_slice_3?
%random_rotation/rotation_matrix/Sin_3Sin$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:?????????2'
%random_rotation/rotation_matrix/Sin_3?
5random_rotation/rotation_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        27
5random_rotation/rotation_matrix/strided_slice_4/stack?
7random_rotation/rotation_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation/rotation_matrix/strided_slice_4/stack_1?
7random_rotation/rotation_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7random_rotation/rotation_matrix/strided_slice_4/stack_2?
/random_rotation/rotation_matrix/strided_slice_4StridedSlice)random_rotation/rotation_matrix/Sin_3:y:0>random_rotation/rotation_matrix/strided_slice_4/stack:output:0@random_rotation/rotation_matrix/strided_slice_4/stack_1:output:0@random_rotation/rotation_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
new_axis_mask21
/random_rotation/rotation_matrix/strided_slice_4?
%random_rotation/rotation_matrix/Cos_3Cos$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:?????????2'
%random_rotation/rotation_matrix/Cos_3?
5random_rotation/rotation_matrix/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        27
5random_rotation/rotation_matrix/strided_slice_5/stack?
7random_rotation/rotation_matrix/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation/rotation_matrix/strided_slice_5/stack_1?
7random_rotation/rotation_matrix/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7random_rotation/rotation_matrix/strided_slice_5/stack_2?
/random_rotation/rotation_matrix/strided_slice_5StridedSlice)random_rotation/rotation_matrix/Cos_3:y:0>random_rotation/rotation_matrix/strided_slice_5/stack:output:0@random_rotation/rotation_matrix/strided_slice_5/stack_1:output:0@random_rotation/rotation_matrix/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
new_axis_mask21
/random_rotation/rotation_matrix/strided_slice_5?
5random_rotation/rotation_matrix/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        27
5random_rotation/rotation_matrix/strided_slice_6/stack?
7random_rotation/rotation_matrix/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation/rotation_matrix/strided_slice_6/stack_1?
7random_rotation/rotation_matrix/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7random_rotation/rotation_matrix/strided_slice_6/stack_2?
/random_rotation/rotation_matrix/strided_slice_6StridedSlice-random_rotation/rotation_matrix/truediv_1:z:0>random_rotation/rotation_matrix/strided_slice_6/stack:output:0@random_rotation/rotation_matrix/strided_slice_6/stack_1:output:0@random_rotation/rotation_matrix/strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
new_axis_mask21
/random_rotation/rotation_matrix/strided_slice_6?
+random_rotation/rotation_matrix/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2-
+random_rotation/rotation_matrix/zeros/mul/y?
)random_rotation/rotation_matrix/zeros/mulMul6random_rotation/rotation_matrix/strided_slice:output:04random_rotation/rotation_matrix/zeros/mul/y:output:0*
T0*
_output_shapes
: 2+
)random_rotation/rotation_matrix/zeros/mul?
,random_rotation/rotation_matrix/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2.
,random_rotation/rotation_matrix/zeros/Less/y?
*random_rotation/rotation_matrix/zeros/LessLess-random_rotation/rotation_matrix/zeros/mul:z:05random_rotation/rotation_matrix/zeros/Less/y:output:0*
T0*
_output_shapes
: 2,
*random_rotation/rotation_matrix/zeros/Less?
.random_rotation/rotation_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :20
.random_rotation/rotation_matrix/zeros/packed/1?
,random_rotation/rotation_matrix/zeros/packedPack6random_rotation/rotation_matrix/strided_slice:output:07random_rotation/rotation_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2.
,random_rotation/rotation_matrix/zeros/packed?
+random_rotation/rotation_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+random_rotation/rotation_matrix/zeros/Const?
%random_rotation/rotation_matrix/zerosFill5random_rotation/rotation_matrix/zeros/packed:output:04random_rotation/rotation_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:?????????2'
%random_rotation/rotation_matrix/zeros?
+random_rotation/rotation_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2-
+random_rotation/rotation_matrix/concat/axis?
&random_rotation/rotation_matrix/concatConcatV28random_rotation/rotation_matrix/strided_slice_1:output:0'random_rotation/rotation_matrix/Neg:y:08random_rotation/rotation_matrix/strided_slice_3:output:08random_rotation/rotation_matrix/strided_slice_4:output:08random_rotation/rotation_matrix/strided_slice_5:output:08random_rotation/rotation_matrix/strided_slice_6:output:0.random_rotation/rotation_matrix/zeros:output:04random_rotation/rotation_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2(
&random_rotation/rotation_matrix/concat?
random_rotation/transform/ShapeShape*random_flip/random_flip_left_right/add:z:0*
T0*
_output_shapes
:2!
random_rotation/transform/Shape?
-random_rotation/transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2/
-random_rotation/transform/strided_slice/stack?
/random_rotation/transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/random_rotation/transform/strided_slice/stack_1?
/random_rotation/transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/random_rotation/transform/strided_slice/stack_2?
'random_rotation/transform/strided_sliceStridedSlice(random_rotation/transform/Shape:output:06random_rotation/transform/strided_slice/stack:output:08random_rotation/transform/strided_slice/stack_1:output:08random_rotation/transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2)
'random_rotation/transform/strided_slice?
$random_rotation/transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2&
$random_rotation/transform/fill_value?
4random_rotation/transform/ImageProjectiveTransformV3ImageProjectiveTransformV3*random_flip/random_flip_left_right/add:z:0/random_rotation/rotation_matrix/concat:output:00random_rotation/transform/strided_slice:output:0-random_rotation/transform/fill_value:output:0*/
_output_shapes
:?????????*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR26
4random_rotation/transform/ImageProjectiveTransformV3?
$random_contrast/random_uniform/shapeConst*
_output_shapes
: *
dtype0*
valueB 2&
$random_contrast/random_uniform/shape?
"random_contrast/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2$
"random_contrast/random_uniform/min?
"random_contrast/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *?̌?2$
"random_contrast/random_uniform/max?
,random_contrast/random_uniform/RandomUniformRandomUniform-random_contrast/random_uniform/shape:output:0*
T0*
_output_shapes
: *
dtype02.
,random_contrast/random_uniform/RandomUniform?
"random_contrast/random_uniform/subSub+random_contrast/random_uniform/max:output:0+random_contrast/random_uniform/min:output:0*
T0*
_output_shapes
: 2$
"random_contrast/random_uniform/sub?
"random_contrast/random_uniform/mulMul5random_contrast/random_uniform/RandomUniform:output:0&random_contrast/random_uniform/sub:z:0*
T0*
_output_shapes
: 2$
"random_contrast/random_uniform/mul?
random_contrast/random_uniformAdd&random_contrast/random_uniform/mul:z:0+random_contrast/random_uniform/min:output:0*
T0*
_output_shapes
: 2 
random_contrast/random_uniform?
random_contrast/adjust_contrastAdjustContrastv2Irandom_rotation/transform/ImageProjectiveTransformV3:transformed_images:0"random_contrast/random_uniform:z:0*/
_output_shapes
:?????????2!
random_contrast/adjust_contrast?
(random_contrast/adjust_contrast/IdentityIdentity(random_contrast/adjust_contrast:output:0*
T0*/
_output_shapes
:?????????2*
(random_contrast/adjust_contrast/Identity?
IdentityIdentity1random_contrast/adjust_contrast/Identity:output:01^random_rotation/stateful_uniform/StatefulUniform*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*2
_input_shapes!
:?????????:2d
0random_rotation/stateful_uniform/StatefulUniform0random_rotation/stateful_uniform/StatefulUniform:b ^
/
_output_shapes
:?????????
+
_user_specified_namerandom_flip_input
?
H
,__inference_flatten_99_layer_call_fn_2512691

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
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_flatten_99_layer_call_and_return_conditional_losses_25119662
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
?
I
-__inference_dropout_110_layer_call_fn_2512680

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
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_dropout_110_layer_call_and_return_conditional_losses_25119472
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
f
-__inference_dropout_110_layer_call_fn_2512675

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
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_dropout_110_layer_call_and_return_conditional_losses_25119422
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
?
g
H__inference_dropout_110_layer_call_and_return_conditional_losses_2511942

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
?$
?
I__inference_Fourth_Model_layer_call_and_return_conditional_losses_2512084$
 training_data_augmentation_input
conv2d_143_2512060
conv2d_143_2512062
dense_308_2512068
dense_308_2512070
dense_309_2512073
dense_309_2512075
dense_310_2512078
dense_310_2512080
identity??"conv2d_143/StatefulPartitionedCall?!dense_308/StatefulPartitionedCall?!dense_309/StatefulPartitionedCall?!dense_310/StatefulPartitionedCall?
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
GPU 2J 8? *`
f[RY
W__inference_Training_Data_Augmentation_layer_call_and_return_conditional_losses_25118682,
*Training_Data_Augmentation/PartitionedCall?
"conv2d_143/StatefulPartitionedCallStatefulPartitionedCall3Training_Data_Augmentation/PartitionedCall:output:0conv2d_143_2512060conv2d_143_2512062*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conv2d_143_layer_call_and_return_conditional_losses_25119132$
"conv2d_143/StatefulPartitionedCall?
!max_pooling2d_132/PartitionedCallPartitionedCall+conv2d_143/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_max_pooling2d_132_layer_call_and_return_conditional_losses_25118772#
!max_pooling2d_132/PartitionedCall?
dropout_110/PartitionedCallPartitionedCall*max_pooling2d_132/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_dropout_110_layer_call_and_return_conditional_losses_25119472
dropout_110/PartitionedCall?
flatten_99/PartitionedCallPartitionedCall$dropout_110/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????** 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_flatten_99_layer_call_and_return_conditional_losses_25119662
flatten_99/PartitionedCall?
!dense_308/StatefulPartitionedCallStatefulPartitionedCall#flatten_99/PartitionedCall:output:0dense_308_2512068dense_308_2512070*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_308_layer_call_and_return_conditional_losses_25119852#
!dense_308/StatefulPartitionedCall?
!dense_309/StatefulPartitionedCallStatefulPartitionedCall*dense_308/StatefulPartitionedCall:output:0dense_309_2512073dense_309_2512075*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_309_layer_call_and_return_conditional_losses_25120122#
!dense_309/StatefulPartitionedCall?
!dense_310/StatefulPartitionedCallStatefulPartitionedCall*dense_309/StatefulPartitionedCall:output:0dense_310_2512078dense_310_2512080*
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
GPU 2J 8? *O
fJRH
F__inference_dense_310_layer_call_and_return_conditional_losses_25120392#
!dense_310/StatefulPartitionedCall?
IdentityIdentity*dense_310/StatefulPartitionedCall:output:0#^conv2d_143/StatefulPartitionedCall"^dense_308/StatefulPartitionedCall"^dense_309/StatefulPartitionedCall"^dense_310/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::2H
"conv2d_143/StatefulPartitionedCall"conv2d_143/StatefulPartitionedCall2F
!dense_308/StatefulPartitionedCall!dense_308/StatefulPartitionedCall2F
!dense_309/StatefulPartitionedCall!dense_309/StatefulPartitionedCall2F
!dense_310/StatefulPartitionedCall!dense_310/StatefulPartitionedCall:q m
/
_output_shapes
:?????????
:
_user_specified_name" Training_Data_Augmentation_input
?	
?
F__inference_dense_310_layer_call_and_return_conditional_losses_2512039

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
?
?
%__inference_signature_wrapper_2512218$
 training_data_augmentation_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall training_data_augmentation_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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

*-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__wrapped_model_25115692
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
StatefulPartitionedCallStatefulPartitionedCall:q m
/
_output_shapes
:?????????
:
_user_specified_name" Training_Data_Augmentation_input
?(
?
I__inference_Fourth_Model_layer_call_and_return_conditional_losses_2512117

inputs&
"training_data_augmentation_2512090
conv2d_143_2512093
conv2d_143_2512095
dense_308_2512101
dense_308_2512103
dense_309_2512106
dense_309_2512108
dense_310_2512111
dense_310_2512113
identity??2Training_Data_Augmentation/StatefulPartitionedCall?"conv2d_143/StatefulPartitionedCall?!dense_308/StatefulPartitionedCall?!dense_309/StatefulPartitionedCall?!dense_310/StatefulPartitionedCall?#dropout_110/StatefulPartitionedCall?
2Training_Data_Augmentation/StatefulPartitionedCallStatefulPartitionedCallinputs"training_data_augmentation_2512090*
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
GPU 2J 8? *`
f[RY
W__inference_Training_Data_Augmentation_layer_call_and_return_conditional_losses_251185724
2Training_Data_Augmentation/StatefulPartitionedCall?
"conv2d_143/StatefulPartitionedCallStatefulPartitionedCall;Training_Data_Augmentation/StatefulPartitionedCall:output:0conv2d_143_2512093conv2d_143_2512095*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conv2d_143_layer_call_and_return_conditional_losses_25119132$
"conv2d_143/StatefulPartitionedCall?
!max_pooling2d_132/PartitionedCallPartitionedCall+conv2d_143/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_max_pooling2d_132_layer_call_and_return_conditional_losses_25118772#
!max_pooling2d_132/PartitionedCall?
#dropout_110/StatefulPartitionedCallStatefulPartitionedCall*max_pooling2d_132/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_dropout_110_layer_call_and_return_conditional_losses_25119422%
#dropout_110/StatefulPartitionedCall?
flatten_99/PartitionedCallPartitionedCall,dropout_110/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????** 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_flatten_99_layer_call_and_return_conditional_losses_25119662
flatten_99/PartitionedCall?
!dense_308/StatefulPartitionedCallStatefulPartitionedCall#flatten_99/PartitionedCall:output:0dense_308_2512101dense_308_2512103*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_308_layer_call_and_return_conditional_losses_25119852#
!dense_308/StatefulPartitionedCall?
!dense_309/StatefulPartitionedCallStatefulPartitionedCall*dense_308/StatefulPartitionedCall:output:0dense_309_2512106dense_309_2512108*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_309_layer_call_and_return_conditional_losses_25120122#
!dense_309/StatefulPartitionedCall?
!dense_310/StatefulPartitionedCallStatefulPartitionedCall*dense_309/StatefulPartitionedCall:output:0dense_310_2512111dense_310_2512113*
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
GPU 2J 8? *O
fJRH
F__inference_dense_310_layer_call_and_return_conditional_losses_25120392#
!dense_310/StatefulPartitionedCall?
IdentityIdentity*dense_310/StatefulPartitionedCall:output:03^Training_Data_Augmentation/StatefulPartitionedCall#^conv2d_143/StatefulPartitionedCall"^dense_308/StatefulPartitionedCall"^dense_309/StatefulPartitionedCall"^dense_310/StatefulPartitionedCall$^dropout_110/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*R
_input_shapesA
?:?????????:::::::::2h
2Training_Data_Augmentation/StatefulPartitionedCall2Training_Data_Augmentation/StatefulPartitionedCall2H
"conv2d_143/StatefulPartitionedCall"conv2d_143/StatefulPartitionedCall2F
!dense_308/StatefulPartitionedCall!dense_308/StatefulPartitionedCall2F
!dense_309/StatefulPartitionedCall!dense_309/StatefulPartitionedCall2F
!dense_310/StatefulPartitionedCall!dense_310/StatefulPartitionedCall2J
#dropout_110/StatefulPartitionedCall#dropout_110/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
.__inference_Fourth_Model_layer_call_fn_2512187$
 training_data_augmentation_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall training_data_augmentation_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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

*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_Fourth_Model_layer_call_and_return_conditional_losses_25121682
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
StatefulPartitionedCallStatefulPartitionedCall:q m
/
_output_shapes
:?????????
:
_user_specified_name" Training_Data_Augmentation_input
?	
?
.__inference_Fourth_Model_layer_call_fn_2512138$
 training_data_augmentation_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall training_data_augmentation_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_Fourth_Model_layer_call_and_return_conditional_losses_25121172
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*R
_input_shapesA
?:?????????:::::::::22
StatefulPartitionedCallStatefulPartitionedCall:q m
/
_output_shapes
:?????????
:
_user_specified_name" Training_Data_Augmentation_input
?
f
H__inference_dropout_110_layer_call_and_return_conditional_losses_2511947

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
?
?
+__inference_dense_308_layer_call_fn_2512711

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
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_308_layer_call_and_return_conditional_losses_25119852
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
?	
?
F__inference_dense_308_layer_call_and_return_conditional_losses_2512702

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
X
<__inference_Training_Data_Augmentation_layer_call_fn_2512633

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
GPU 2J 8? *`
f[RY
W__inference_Training_Data_Augmentation_layer_call_and_return_conditional_losses_25118682
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
??
?
W__inference_Training_Data_Augmentation_layer_call_and_return_conditional_losses_2512617

inputs=
9random_rotation_stateful_uniform_statefuluniform_resource
identity??0random_rotation/stateful_uniform/StatefulUniform?
5random_flip/random_flip_left_right/control_dependencyIdentityinputs*
T0*
_class
loc:@inputs*/
_output_shapes
:?????????27
5random_flip/random_flip_left_right/control_dependency?
(random_flip/random_flip_left_right/ShapeShape>random_flip/random_flip_left_right/control_dependency:output:0*
T0*
_output_shapes
:2*
(random_flip/random_flip_left_right/Shape?
6random_flip/random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 28
6random_flip/random_flip_left_right/strided_slice/stack?
8random_flip/random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8random_flip/random_flip_left_right/strided_slice/stack_1?
8random_flip/random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8random_flip/random_flip_left_right/strided_slice/stack_2?
0random_flip/random_flip_left_right/strided_sliceStridedSlice1random_flip/random_flip_left_right/Shape:output:0?random_flip/random_flip_left_right/strided_slice/stack:output:0Arandom_flip/random_flip_left_right/strided_slice/stack_1:output:0Arandom_flip/random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask22
0random_flip/random_flip_left_right/strided_slice?
7random_flip/random_flip_left_right/random_uniform/shapePack9random_flip/random_flip_left_right/strided_slice:output:0*
N*
T0*
_output_shapes
:29
7random_flip/random_flip_left_right/random_uniform/shape?
5random_flip/random_flip_left_right/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    27
5random_flip/random_flip_left_right/random_uniform/min?
5random_flip/random_flip_left_right/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ??27
5random_flip/random_flip_left_right/random_uniform/max?
?random_flip/random_flip_left_right/random_uniform/RandomUniformRandomUniform@random_flip/random_flip_left_right/random_uniform/shape:output:0*
T0*#
_output_shapes
:?????????*
dtype02A
?random_flip/random_flip_left_right/random_uniform/RandomUniform?
5random_flip/random_flip_left_right/random_uniform/MulMulHrandom_flip/random_flip_left_right/random_uniform/RandomUniform:output:0>random_flip/random_flip_left_right/random_uniform/max:output:0*
T0*#
_output_shapes
:?????????27
5random_flip/random_flip_left_right/random_uniform/Mul?
2random_flip/random_flip_left_right/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :24
2random_flip/random_flip_left_right/Reshape/shape/1?
2random_flip/random_flip_left_right/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :24
2random_flip/random_flip_left_right/Reshape/shape/2?
2random_flip/random_flip_left_right/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :24
2random_flip/random_flip_left_right/Reshape/shape/3?
0random_flip/random_flip_left_right/Reshape/shapePack9random_flip/random_flip_left_right/strided_slice:output:0;random_flip/random_flip_left_right/Reshape/shape/1:output:0;random_flip/random_flip_left_right/Reshape/shape/2:output:0;random_flip/random_flip_left_right/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:22
0random_flip/random_flip_left_right/Reshape/shape?
*random_flip/random_flip_left_right/ReshapeReshape9random_flip/random_flip_left_right/random_uniform/Mul:z:09random_flip/random_flip_left_right/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2,
*random_flip/random_flip_left_right/Reshape?
(random_flip/random_flip_left_right/RoundRound3random_flip/random_flip_left_right/Reshape:output:0*
T0*/
_output_shapes
:?????????2*
(random_flip/random_flip_left_right/Round?
1random_flip/random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:23
1random_flip/random_flip_left_right/ReverseV2/axis?
,random_flip/random_flip_left_right/ReverseV2	ReverseV2>random_flip/random_flip_left_right/control_dependency:output:0:random_flip/random_flip_left_right/ReverseV2/axis:output:0*
T0*/
_output_shapes
:?????????2.
,random_flip/random_flip_left_right/ReverseV2?
&random_flip/random_flip_left_right/mulMul,random_flip/random_flip_left_right/Round:y:05random_flip/random_flip_left_right/ReverseV2:output:0*
T0*/
_output_shapes
:?????????2(
&random_flip/random_flip_left_right/mul?
(random_flip/random_flip_left_right/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2*
(random_flip/random_flip_left_right/sub/x?
&random_flip/random_flip_left_right/subSub1random_flip/random_flip_left_right/sub/x:output:0,random_flip/random_flip_left_right/Round:y:0*
T0*/
_output_shapes
:?????????2(
&random_flip/random_flip_left_right/sub?
(random_flip/random_flip_left_right/mul_1Mul*random_flip/random_flip_left_right/sub:z:0>random_flip/random_flip_left_right/control_dependency:output:0*
T0*/
_output_shapes
:?????????2*
(random_flip/random_flip_left_right/mul_1?
&random_flip/random_flip_left_right/addAddV2*random_flip/random_flip_left_right/mul:z:0,random_flip/random_flip_left_right/mul_1:z:0*
T0*/
_output_shapes
:?????????2(
&random_flip/random_flip_left_right/add?
random_rotation/ShapeShape*random_flip/random_flip_left_right/add:z:0*
T0*
_output_shapes
:2
random_rotation/Shape?
#random_rotation/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#random_rotation/strided_slice/stack?
%random_rotation/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%random_rotation/strided_slice/stack_1?
%random_rotation/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%random_rotation/strided_slice/stack_2?
random_rotation/strided_sliceStridedSlicerandom_rotation/Shape:output:0,random_rotation/strided_slice/stack:output:0.random_rotation/strided_slice/stack_1:output:0.random_rotation/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
random_rotation/strided_slice?
%random_rotation/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2'
%random_rotation/strided_slice_1/stack?
'random_rotation/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'random_rotation/strided_slice_1/stack_1?
'random_rotation/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'random_rotation/strided_slice_1/stack_2?
random_rotation/strided_slice_1StridedSlicerandom_rotation/Shape:output:0.random_rotation/strided_slice_1/stack:output:00random_rotation/strided_slice_1/stack_1:output:00random_rotation/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
random_rotation/strided_slice_1?
random_rotation/CastCast(random_rotation/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
random_rotation/Cast?
%random_rotation/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2'
%random_rotation/strided_slice_2/stack?
'random_rotation/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'random_rotation/strided_slice_2/stack_1?
'random_rotation/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'random_rotation/strided_slice_2/stack_2?
random_rotation/strided_slice_2StridedSlicerandom_rotation/Shape:output:0.random_rotation/strided_slice_2/stack:output:00random_rotation/strided_slice_2/stack_1:output:00random_rotation/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
random_rotation/strided_slice_2?
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
&random_rotation/stateful_uniform/shape?
$random_rotation/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *|٠?2&
$random_rotation/stateful_uniform/min?
$random_rotation/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *|٠?2&
$random_rotation/stateful_uniform/max?
:random_rotation/stateful_uniform/StatefulUniform/algorithmConst*
_output_shapes
: *
dtype0	*
value	B	 R2<
:random_rotation/stateful_uniform/StatefulUniform/algorithm?
0random_rotation/stateful_uniform/StatefulUniformStatefulUniform9random_rotation_stateful_uniform_statefuluniform_resourceCrandom_rotation/stateful_uniform/StatefulUniform/algorithm:output:0/random_rotation/stateful_uniform/shape:output:0*#
_output_shapes
:?????????*
shape_dtype022
0random_rotation/stateful_uniform/StatefulUniform?
$random_rotation/stateful_uniform/subSub-random_rotation/stateful_uniform/max:output:0-random_rotation/stateful_uniform/min:output:0*
T0*
_output_shapes
: 2&
$random_rotation/stateful_uniform/sub?
$random_rotation/stateful_uniform/mulMul9random_rotation/stateful_uniform/StatefulUniform:output:0(random_rotation/stateful_uniform/sub:z:0*
T0*#
_output_shapes
:?????????2&
$random_rotation/stateful_uniform/mul?
 random_rotation/stateful_uniformAdd(random_rotation/stateful_uniform/mul:z:0-random_rotation/stateful_uniform/min:output:0*
T0*#
_output_shapes
:?????????2"
 random_rotation/stateful_uniform?
%random_rotation/rotation_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2'
%random_rotation/rotation_matrix/sub/y?
#random_rotation/rotation_matrix/subSubrandom_rotation/Cast_1:y:0.random_rotation/rotation_matrix/sub/y:output:0*
T0*
_output_shapes
: 2%
#random_rotation/rotation_matrix/sub?
#random_rotation/rotation_matrix/CosCos$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:?????????2%
#random_rotation/rotation_matrix/Cos?
'random_rotation/rotation_matrix/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2)
'random_rotation/rotation_matrix/sub_1/y?
%random_rotation/rotation_matrix/sub_1Subrandom_rotation/Cast_1:y:00random_rotation/rotation_matrix/sub_1/y:output:0*
T0*
_output_shapes
: 2'
%random_rotation/rotation_matrix/sub_1?
#random_rotation/rotation_matrix/mulMul'random_rotation/rotation_matrix/Cos:y:0)random_rotation/rotation_matrix/sub_1:z:0*
T0*#
_output_shapes
:?????????2%
#random_rotation/rotation_matrix/mul?
#random_rotation/rotation_matrix/SinSin$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:?????????2%
#random_rotation/rotation_matrix/Sin?
'random_rotation/rotation_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2)
'random_rotation/rotation_matrix/sub_2/y?
%random_rotation/rotation_matrix/sub_2Subrandom_rotation/Cast:y:00random_rotation/rotation_matrix/sub_2/y:output:0*
T0*
_output_shapes
: 2'
%random_rotation/rotation_matrix/sub_2?
%random_rotation/rotation_matrix/mul_1Mul'random_rotation/rotation_matrix/Sin:y:0)random_rotation/rotation_matrix/sub_2:z:0*
T0*#
_output_shapes
:?????????2'
%random_rotation/rotation_matrix/mul_1?
%random_rotation/rotation_matrix/sub_3Sub'random_rotation/rotation_matrix/mul:z:0)random_rotation/rotation_matrix/mul_1:z:0*
T0*#
_output_shapes
:?????????2'
%random_rotation/rotation_matrix/sub_3?
%random_rotation/rotation_matrix/sub_4Sub'random_rotation/rotation_matrix/sub:z:0)random_rotation/rotation_matrix/sub_3:z:0*
T0*#
_output_shapes
:?????????2'
%random_rotation/rotation_matrix/sub_4?
)random_rotation/rotation_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2+
)random_rotation/rotation_matrix/truediv/y?
'random_rotation/rotation_matrix/truedivRealDiv)random_rotation/rotation_matrix/sub_4:z:02random_rotation/rotation_matrix/truediv/y:output:0*
T0*#
_output_shapes
:?????????2)
'random_rotation/rotation_matrix/truediv?
'random_rotation/rotation_matrix/sub_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2)
'random_rotation/rotation_matrix/sub_5/y?
%random_rotation/rotation_matrix/sub_5Subrandom_rotation/Cast:y:00random_rotation/rotation_matrix/sub_5/y:output:0*
T0*
_output_shapes
: 2'
%random_rotation/rotation_matrix/sub_5?
%random_rotation/rotation_matrix/Sin_1Sin$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:?????????2'
%random_rotation/rotation_matrix/Sin_1?
'random_rotation/rotation_matrix/sub_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2)
'random_rotation/rotation_matrix/sub_6/y?
%random_rotation/rotation_matrix/sub_6Subrandom_rotation/Cast_1:y:00random_rotation/rotation_matrix/sub_6/y:output:0*
T0*
_output_shapes
: 2'
%random_rotation/rotation_matrix/sub_6?
%random_rotation/rotation_matrix/mul_2Mul)random_rotation/rotation_matrix/Sin_1:y:0)random_rotation/rotation_matrix/sub_6:z:0*
T0*#
_output_shapes
:?????????2'
%random_rotation/rotation_matrix/mul_2?
%random_rotation/rotation_matrix/Cos_1Cos$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:?????????2'
%random_rotation/rotation_matrix/Cos_1?
'random_rotation/rotation_matrix/sub_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2)
'random_rotation/rotation_matrix/sub_7/y?
%random_rotation/rotation_matrix/sub_7Subrandom_rotation/Cast:y:00random_rotation/rotation_matrix/sub_7/y:output:0*
T0*
_output_shapes
: 2'
%random_rotation/rotation_matrix/sub_7?
%random_rotation/rotation_matrix/mul_3Mul)random_rotation/rotation_matrix/Cos_1:y:0)random_rotation/rotation_matrix/sub_7:z:0*
T0*#
_output_shapes
:?????????2'
%random_rotation/rotation_matrix/mul_3?
#random_rotation/rotation_matrix/addAddV2)random_rotation/rotation_matrix/mul_2:z:0)random_rotation/rotation_matrix/mul_3:z:0*
T0*#
_output_shapes
:?????????2%
#random_rotation/rotation_matrix/add?
%random_rotation/rotation_matrix/sub_8Sub)random_rotation/rotation_matrix/sub_5:z:0'random_rotation/rotation_matrix/add:z:0*
T0*#
_output_shapes
:?????????2'
%random_rotation/rotation_matrix/sub_8?
+random_rotation/rotation_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2-
+random_rotation/rotation_matrix/truediv_1/y?
)random_rotation/rotation_matrix/truediv_1RealDiv)random_rotation/rotation_matrix/sub_8:z:04random_rotation/rotation_matrix/truediv_1/y:output:0*
T0*#
_output_shapes
:?????????2+
)random_rotation/rotation_matrix/truediv_1?
%random_rotation/rotation_matrix/ShapeShape$random_rotation/stateful_uniform:z:0*
T0*
_output_shapes
:2'
%random_rotation/rotation_matrix/Shape?
3random_rotation/rotation_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3random_rotation/rotation_matrix/strided_slice/stack?
5random_rotation/rotation_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5random_rotation/rotation_matrix/strided_slice/stack_1?
5random_rotation/rotation_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5random_rotation/rotation_matrix/strided_slice/stack_2?
-random_rotation/rotation_matrix/strided_sliceStridedSlice.random_rotation/rotation_matrix/Shape:output:0<random_rotation/rotation_matrix/strided_slice/stack:output:0>random_rotation/rotation_matrix/strided_slice/stack_1:output:0>random_rotation/rotation_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-random_rotation/rotation_matrix/strided_slice?
%random_rotation/rotation_matrix/Cos_2Cos$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:?????????2'
%random_rotation/rotation_matrix/Cos_2?
5random_rotation/rotation_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        27
5random_rotation/rotation_matrix/strided_slice_1/stack?
7random_rotation/rotation_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation/rotation_matrix/strided_slice_1/stack_1?
7random_rotation/rotation_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7random_rotation/rotation_matrix/strided_slice_1/stack_2?
/random_rotation/rotation_matrix/strided_slice_1StridedSlice)random_rotation/rotation_matrix/Cos_2:y:0>random_rotation/rotation_matrix/strided_slice_1/stack:output:0@random_rotation/rotation_matrix/strided_slice_1/stack_1:output:0@random_rotation/rotation_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
new_axis_mask21
/random_rotation/rotation_matrix/strided_slice_1?
%random_rotation/rotation_matrix/Sin_2Sin$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:?????????2'
%random_rotation/rotation_matrix/Sin_2?
5random_rotation/rotation_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        27
5random_rotation/rotation_matrix/strided_slice_2/stack?
7random_rotation/rotation_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation/rotation_matrix/strided_slice_2/stack_1?
7random_rotation/rotation_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7random_rotation/rotation_matrix/strided_slice_2/stack_2?
/random_rotation/rotation_matrix/strided_slice_2StridedSlice)random_rotation/rotation_matrix/Sin_2:y:0>random_rotation/rotation_matrix/strided_slice_2/stack:output:0@random_rotation/rotation_matrix/strided_slice_2/stack_1:output:0@random_rotation/rotation_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
new_axis_mask21
/random_rotation/rotation_matrix/strided_slice_2?
#random_rotation/rotation_matrix/NegNeg8random_rotation/rotation_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????2%
#random_rotation/rotation_matrix/Neg?
5random_rotation/rotation_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        27
5random_rotation/rotation_matrix/strided_slice_3/stack?
7random_rotation/rotation_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation/rotation_matrix/strided_slice_3/stack_1?
7random_rotation/rotation_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7random_rotation/rotation_matrix/strided_slice_3/stack_2?
/random_rotation/rotation_matrix/strided_slice_3StridedSlice+random_rotation/rotation_matrix/truediv:z:0>random_rotation/rotation_matrix/strided_slice_3/stack:output:0@random_rotation/rotation_matrix/strided_slice_3/stack_1:output:0@random_rotation/rotation_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
new_axis_mask21
/random_rotation/rotation_matrix/strided_slice_3?
%random_rotation/rotation_matrix/Sin_3Sin$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:?????????2'
%random_rotation/rotation_matrix/Sin_3?
5random_rotation/rotation_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        27
5random_rotation/rotation_matrix/strided_slice_4/stack?
7random_rotation/rotation_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation/rotation_matrix/strided_slice_4/stack_1?
7random_rotation/rotation_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7random_rotation/rotation_matrix/strided_slice_4/stack_2?
/random_rotation/rotation_matrix/strided_slice_4StridedSlice)random_rotation/rotation_matrix/Sin_3:y:0>random_rotation/rotation_matrix/strided_slice_4/stack:output:0@random_rotation/rotation_matrix/strided_slice_4/stack_1:output:0@random_rotation/rotation_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
new_axis_mask21
/random_rotation/rotation_matrix/strided_slice_4?
%random_rotation/rotation_matrix/Cos_3Cos$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:?????????2'
%random_rotation/rotation_matrix/Cos_3?
5random_rotation/rotation_matrix/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        27
5random_rotation/rotation_matrix/strided_slice_5/stack?
7random_rotation/rotation_matrix/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation/rotation_matrix/strided_slice_5/stack_1?
7random_rotation/rotation_matrix/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7random_rotation/rotation_matrix/strided_slice_5/stack_2?
/random_rotation/rotation_matrix/strided_slice_5StridedSlice)random_rotation/rotation_matrix/Cos_3:y:0>random_rotation/rotation_matrix/strided_slice_5/stack:output:0@random_rotation/rotation_matrix/strided_slice_5/stack_1:output:0@random_rotation/rotation_matrix/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
new_axis_mask21
/random_rotation/rotation_matrix/strided_slice_5?
5random_rotation/rotation_matrix/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        27
5random_rotation/rotation_matrix/strided_slice_6/stack?
7random_rotation/rotation_matrix/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation/rotation_matrix/strided_slice_6/stack_1?
7random_rotation/rotation_matrix/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7random_rotation/rotation_matrix/strided_slice_6/stack_2?
/random_rotation/rotation_matrix/strided_slice_6StridedSlice-random_rotation/rotation_matrix/truediv_1:z:0>random_rotation/rotation_matrix/strided_slice_6/stack:output:0@random_rotation/rotation_matrix/strided_slice_6/stack_1:output:0@random_rotation/rotation_matrix/strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*

begin_mask*
end_mask*
new_axis_mask21
/random_rotation/rotation_matrix/strided_slice_6?
+random_rotation/rotation_matrix/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2-
+random_rotation/rotation_matrix/zeros/mul/y?
)random_rotation/rotation_matrix/zeros/mulMul6random_rotation/rotation_matrix/strided_slice:output:04random_rotation/rotation_matrix/zeros/mul/y:output:0*
T0*
_output_shapes
: 2+
)random_rotation/rotation_matrix/zeros/mul?
,random_rotation/rotation_matrix/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2.
,random_rotation/rotation_matrix/zeros/Less/y?
*random_rotation/rotation_matrix/zeros/LessLess-random_rotation/rotation_matrix/zeros/mul:z:05random_rotation/rotation_matrix/zeros/Less/y:output:0*
T0*
_output_shapes
: 2,
*random_rotation/rotation_matrix/zeros/Less?
.random_rotation/rotation_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :20
.random_rotation/rotation_matrix/zeros/packed/1?
,random_rotation/rotation_matrix/zeros/packedPack6random_rotation/rotation_matrix/strided_slice:output:07random_rotation/rotation_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2.
,random_rotation/rotation_matrix/zeros/packed?
+random_rotation/rotation_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+random_rotation/rotation_matrix/zeros/Const?
%random_rotation/rotation_matrix/zerosFill5random_rotation/rotation_matrix/zeros/packed:output:04random_rotation/rotation_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:?????????2'
%random_rotation/rotation_matrix/zeros?
+random_rotation/rotation_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2-
+random_rotation/rotation_matrix/concat/axis?
&random_rotation/rotation_matrix/concatConcatV28random_rotation/rotation_matrix/strided_slice_1:output:0'random_rotation/rotation_matrix/Neg:y:08random_rotation/rotation_matrix/strided_slice_3:output:08random_rotation/rotation_matrix/strided_slice_4:output:08random_rotation/rotation_matrix/strided_slice_5:output:08random_rotation/rotation_matrix/strided_slice_6:output:0.random_rotation/rotation_matrix/zeros:output:04random_rotation/rotation_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2(
&random_rotation/rotation_matrix/concat?
random_rotation/transform/ShapeShape*random_flip/random_flip_left_right/add:z:0*
T0*
_output_shapes
:2!
random_rotation/transform/Shape?
-random_rotation/transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2/
-random_rotation/transform/strided_slice/stack?
/random_rotation/transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/random_rotation/transform/strided_slice/stack_1?
/random_rotation/transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/random_rotation/transform/strided_slice/stack_2?
'random_rotation/transform/strided_sliceStridedSlice(random_rotation/transform/Shape:output:06random_rotation/transform/strided_slice/stack:output:08random_rotation/transform/strided_slice/stack_1:output:08random_rotation/transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2)
'random_rotation/transform/strided_slice?
$random_rotation/transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2&
$random_rotation/transform/fill_value?
4random_rotation/transform/ImageProjectiveTransformV3ImageProjectiveTransformV3*random_flip/random_flip_left_right/add:z:0/random_rotation/rotation_matrix/concat:output:00random_rotation/transform/strided_slice:output:0-random_rotation/transform/fill_value:output:0*/
_output_shapes
:?????????*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR26
4random_rotation/transform/ImageProjectiveTransformV3?
$random_contrast/random_uniform/shapeConst*
_output_shapes
: *
dtype0*
valueB 2&
$random_contrast/random_uniform/shape?
"random_contrast/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2$
"random_contrast/random_uniform/min?
"random_contrast/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *?̌?2$
"random_contrast/random_uniform/max?
,random_contrast/random_uniform/RandomUniformRandomUniform-random_contrast/random_uniform/shape:output:0*
T0*
_output_shapes
: *
dtype02.
,random_contrast/random_uniform/RandomUniform?
"random_contrast/random_uniform/subSub+random_contrast/random_uniform/max:output:0+random_contrast/random_uniform/min:output:0*
T0*
_output_shapes
: 2$
"random_contrast/random_uniform/sub?
"random_contrast/random_uniform/mulMul5random_contrast/random_uniform/RandomUniform:output:0&random_contrast/random_uniform/sub:z:0*
T0*
_output_shapes
: 2$
"random_contrast/random_uniform/mul?
random_contrast/random_uniformAdd&random_contrast/random_uniform/mul:z:0+random_contrast/random_uniform/min:output:0*
T0*
_output_shapes
: 2 
random_contrast/random_uniform?
random_contrast/adjust_contrastAdjustContrastv2Irandom_rotation/transform/ImageProjectiveTransformV3:transformed_images:0"random_contrast/random_uniform:z:0*/
_output_shapes
:?????????2!
random_contrast/adjust_contrast?
(random_contrast/adjust_contrast/IdentityIdentity(random_contrast/adjust_contrast:output:0*
T0*/
_output_shapes
:?????????2*
(random_contrast/adjust_contrast/Identity?
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
?	
?
F__inference_dense_309_layer_call_and_return_conditional_losses_2512722

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
?:
?
"__inference__wrapped_model_2511569$
 training_data_augmentation_input:
6fourth_model_conv2d_143_conv2d_readvariableop_resource;
7fourth_model_conv2d_143_biasadd_readvariableop_resource9
5fourth_model_dense_308_matmul_readvariableop_resource:
6fourth_model_dense_308_biasadd_readvariableop_resource9
5fourth_model_dense_309_matmul_readvariableop_resource:
6fourth_model_dense_309_biasadd_readvariableop_resource9
5fourth_model_dense_310_matmul_readvariableop_resource:
6fourth_model_dense_310_biasadd_readvariableop_resource
identity??.Fourth_Model/conv2d_143/BiasAdd/ReadVariableOp?-Fourth_Model/conv2d_143/Conv2D/ReadVariableOp?-Fourth_Model/dense_308/BiasAdd/ReadVariableOp?,Fourth_Model/dense_308/MatMul/ReadVariableOp?-Fourth_Model/dense_309/BiasAdd/ReadVariableOp?,Fourth_Model/dense_309/MatMul/ReadVariableOp?-Fourth_Model/dense_310/BiasAdd/ReadVariableOp?,Fourth_Model/dense_310/MatMul/ReadVariableOp?
-Fourth_Model/conv2d_143/Conv2D/ReadVariableOpReadVariableOp6fourth_model_conv2d_143_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02/
-Fourth_Model/conv2d_143/Conv2D/ReadVariableOp?
Fourth_Model/conv2d_143/Conv2DConv2D training_data_augmentation_input5Fourth_Model/conv2d_143/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2 
Fourth_Model/conv2d_143/Conv2D?
.Fourth_Model/conv2d_143/BiasAdd/ReadVariableOpReadVariableOp7fourth_model_conv2d_143_biasadd_readvariableop_resource*
_output_shapes
: *
dtype020
.Fourth_Model/conv2d_143/BiasAdd/ReadVariableOp?
Fourth_Model/conv2d_143/BiasAddBiasAdd'Fourth_Model/conv2d_143/Conv2D:output:06Fourth_Model/conv2d_143/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2!
Fourth_Model/conv2d_143/BiasAdd?
Fourth_Model/conv2d_143/ReluRelu(Fourth_Model/conv2d_143/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
Fourth_Model/conv2d_143/Relu?
&Fourth_Model/max_pooling2d_132/MaxPoolMaxPool*Fourth_Model/conv2d_143/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2(
&Fourth_Model/max_pooling2d_132/MaxPool?
!Fourth_Model/dropout_110/IdentityIdentity/Fourth_Model/max_pooling2d_132/MaxPool:output:0*
T0*/
_output_shapes
:????????? 2#
!Fourth_Model/dropout_110/Identity?
Fourth_Model/flatten_99/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Fourth_Model/flatten_99/Const?
Fourth_Model/flatten_99/ReshapeReshape*Fourth_Model/dropout_110/Identity:output:0&Fourth_Model/flatten_99/Const:output:0*
T0*(
_output_shapes
:??????????*2!
Fourth_Model/flatten_99/Reshape?
,Fourth_Model/dense_308/MatMul/ReadVariableOpReadVariableOp5fourth_model_dense_308_matmul_readvariableop_resource* 
_output_shapes
:
?*?*
dtype02.
,Fourth_Model/dense_308/MatMul/ReadVariableOp?
Fourth_Model/dense_308/MatMulMatMul(Fourth_Model/flatten_99/Reshape:output:04Fourth_Model/dense_308/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Fourth_Model/dense_308/MatMul?
-Fourth_Model/dense_308/BiasAdd/ReadVariableOpReadVariableOp6fourth_model_dense_308_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-Fourth_Model/dense_308/BiasAdd/ReadVariableOp?
Fourth_Model/dense_308/BiasAddBiasAdd'Fourth_Model/dense_308/MatMul:product:05Fourth_Model/dense_308/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
Fourth_Model/dense_308/BiasAdd?
Fourth_Model/dense_308/ReluRelu'Fourth_Model/dense_308/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Fourth_Model/dense_308/Relu?
,Fourth_Model/dense_309/MatMul/ReadVariableOpReadVariableOp5fourth_model_dense_309_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02.
,Fourth_Model/dense_309/MatMul/ReadVariableOp?
Fourth_Model/dense_309/MatMulMatMul)Fourth_Model/dense_308/Relu:activations:04Fourth_Model/dense_309/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Fourth_Model/dense_309/MatMul?
-Fourth_Model/dense_309/BiasAdd/ReadVariableOpReadVariableOp6fourth_model_dense_309_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-Fourth_Model/dense_309/BiasAdd/ReadVariableOp?
Fourth_Model/dense_309/BiasAddBiasAdd'Fourth_Model/dense_309/MatMul:product:05Fourth_Model/dense_309/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
Fourth_Model/dense_309/BiasAdd?
Fourth_Model/dense_309/ReluRelu'Fourth_Model/dense_309/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Fourth_Model/dense_309/Relu?
,Fourth_Model/dense_310/MatMul/ReadVariableOpReadVariableOp5fourth_model_dense_310_matmul_readvariableop_resource*
_output_shapes
:	?
*
dtype02.
,Fourth_Model/dense_310/MatMul/ReadVariableOp?
Fourth_Model/dense_310/MatMulMatMul)Fourth_Model/dense_309/Relu:activations:04Fourth_Model/dense_310/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
Fourth_Model/dense_310/MatMul?
-Fourth_Model/dense_310/BiasAdd/ReadVariableOpReadVariableOp6fourth_model_dense_310_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02/
-Fourth_Model/dense_310/BiasAdd/ReadVariableOp?
Fourth_Model/dense_310/BiasAddBiasAdd'Fourth_Model/dense_310/MatMul:product:05Fourth_Model/dense_310/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2 
Fourth_Model/dense_310/BiasAdd?
Fourth_Model/dense_310/SoftmaxSoftmax'Fourth_Model/dense_310/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2 
Fourth_Model/dense_310/Softmax?
IdentityIdentity(Fourth_Model/dense_310/Softmax:softmax:0/^Fourth_Model/conv2d_143/BiasAdd/ReadVariableOp.^Fourth_Model/conv2d_143/Conv2D/ReadVariableOp.^Fourth_Model/dense_308/BiasAdd/ReadVariableOp-^Fourth_Model/dense_308/MatMul/ReadVariableOp.^Fourth_Model/dense_309/BiasAdd/ReadVariableOp-^Fourth_Model/dense_309/MatMul/ReadVariableOp.^Fourth_Model/dense_310/BiasAdd/ReadVariableOp-^Fourth_Model/dense_310/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::2`
.Fourth_Model/conv2d_143/BiasAdd/ReadVariableOp.Fourth_Model/conv2d_143/BiasAdd/ReadVariableOp2^
-Fourth_Model/conv2d_143/Conv2D/ReadVariableOp-Fourth_Model/conv2d_143/Conv2D/ReadVariableOp2^
-Fourth_Model/dense_308/BiasAdd/ReadVariableOp-Fourth_Model/dense_308/BiasAdd/ReadVariableOp2\
,Fourth_Model/dense_308/MatMul/ReadVariableOp,Fourth_Model/dense_308/MatMul/ReadVariableOp2^
-Fourth_Model/dense_309/BiasAdd/ReadVariableOp-Fourth_Model/dense_309/BiasAdd/ReadVariableOp2\
,Fourth_Model/dense_309/MatMul/ReadVariableOp,Fourth_Model/dense_309/MatMul/ReadVariableOp2^
-Fourth_Model/dense_310/BiasAdd/ReadVariableOp-Fourth_Model/dense_310/BiasAdd/ReadVariableOp2\
,Fourth_Model/dense_310/MatMul/ReadVariableOp,Fourth_Model/dense_310/MatMul/ReadVariableOp:q m
/
_output_shapes
:?????????
:
_user_specified_name" Training_Data_Augmentation_input
?

?
G__inference_conv2d_143_layer_call_and_return_conditional_losses_2511913

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
?
?
,__inference_conv2d_143_layer_call_fn_2512653

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
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conv2d_143_layer_call_and_return_conditional_losses_25119132
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
?
j
N__inference_max_pooling2d_132_layer_call_and_return_conditional_losses_2511877

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
?(
?
I__inference_Fourth_Model_layer_call_and_return_conditional_losses_2512056$
 training_data_augmentation_input&
"training_data_augmentation_2511899
conv2d_143_2511924
conv2d_143_2511926
dense_308_2511996
dense_308_2511998
dense_309_2512023
dense_309_2512025
dense_310_2512050
dense_310_2512052
identity??2Training_Data_Augmentation/StatefulPartitionedCall?"conv2d_143/StatefulPartitionedCall?!dense_308/StatefulPartitionedCall?!dense_309/StatefulPartitionedCall?!dense_310/StatefulPartitionedCall?#dropout_110/StatefulPartitionedCall?
2Training_Data_Augmentation/StatefulPartitionedCallStatefulPartitionedCall training_data_augmentation_input"training_data_augmentation_2511899*
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
GPU 2J 8? *`
f[RY
W__inference_Training_Data_Augmentation_layer_call_and_return_conditional_losses_251185724
2Training_Data_Augmentation/StatefulPartitionedCall?
"conv2d_143/StatefulPartitionedCallStatefulPartitionedCall;Training_Data_Augmentation/StatefulPartitionedCall:output:0conv2d_143_2511924conv2d_143_2511926*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conv2d_143_layer_call_and_return_conditional_losses_25119132$
"conv2d_143/StatefulPartitionedCall?
!max_pooling2d_132/PartitionedCallPartitionedCall+conv2d_143/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_max_pooling2d_132_layer_call_and_return_conditional_losses_25118772#
!max_pooling2d_132/PartitionedCall?
#dropout_110/StatefulPartitionedCallStatefulPartitionedCall*max_pooling2d_132/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_dropout_110_layer_call_and_return_conditional_losses_25119422%
#dropout_110/StatefulPartitionedCall?
flatten_99/PartitionedCallPartitionedCall,dropout_110/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????** 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_flatten_99_layer_call_and_return_conditional_losses_25119662
flatten_99/PartitionedCall?
!dense_308/StatefulPartitionedCallStatefulPartitionedCall#flatten_99/PartitionedCall:output:0dense_308_2511996dense_308_2511998*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_308_layer_call_and_return_conditional_losses_25119852#
!dense_308/StatefulPartitionedCall?
!dense_309/StatefulPartitionedCallStatefulPartitionedCall*dense_308/StatefulPartitionedCall:output:0dense_309_2512023dense_309_2512025*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_309_layer_call_and_return_conditional_losses_25120122#
!dense_309/StatefulPartitionedCall?
!dense_310/StatefulPartitionedCallStatefulPartitionedCall*dense_309/StatefulPartitionedCall:output:0dense_310_2512050dense_310_2512052*
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
GPU 2J 8? *O
fJRH
F__inference_dense_310_layer_call_and_return_conditional_losses_25120392#
!dense_310/StatefulPartitionedCall?
IdentityIdentity*dense_310/StatefulPartitionedCall:output:03^Training_Data_Augmentation/StatefulPartitionedCall#^conv2d_143/StatefulPartitionedCall"^dense_308/StatefulPartitionedCall"^dense_309/StatefulPartitionedCall"^dense_310/StatefulPartitionedCall$^dropout_110/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*R
_input_shapesA
?:?????????:::::::::2h
2Training_Data_Augmentation/StatefulPartitionedCall2Training_Data_Augmentation/StatefulPartitionedCall2H
"conv2d_143/StatefulPartitionedCall"conv2d_143/StatefulPartitionedCall2F
!dense_308/StatefulPartitionedCall!dense_308/StatefulPartitionedCall2F
!dense_309/StatefulPartitionedCall!dense_309/StatefulPartitionedCall2F
!dense_310/StatefulPartitionedCall!dense_310/StatefulPartitionedCall2J
#dropout_110/StatefulPartitionedCall#dropout_110/StatefulPartitionedCall:q m
/
_output_shapes
:?????????
:
_user_specified_name" Training_Data_Augmentation_input
?
?
+__inference_dense_310_layer_call_fn_2512751

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
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_310_layer_call_and_return_conditional_losses_25120392
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
?
?
.__inference_Fourth_Model_layer_call_fn_2512456

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_Fourth_Model_layer_call_and_return_conditional_losses_25121172
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*R
_input_shapesA
?:?????????:::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
c
G__inference_flatten_99_layer_call_and_return_conditional_losses_2511966

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
?
?
+__inference_dense_309_layer_call_fn_2512731

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
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_309_layer_call_and_return_conditional_losses_25120122
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
?	
?
F__inference_dense_310_layer_call_and_return_conditional_losses_2512742

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
?K
?
 __inference__traced_save_2512889
file_prefix0
,savev2_conv2d_143_kernel_read_readvariableop.
*savev2_conv2d_143_bias_read_readvariableop/
+savev2_dense_308_kernel_read_readvariableop-
)savev2_dense_308_bias_read_readvariableop/
+savev2_dense_309_kernel_read_readvariableop-
)savev2_dense_309_bias_read_readvariableop/
+savev2_dense_310_kernel_read_readvariableop-
)savev2_dense_310_bias_read_readvariableop(
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
"savev2_count_1_read_readvariableop7
3savev2_adam_conv2d_143_kernel_m_read_readvariableop5
1savev2_adam_conv2d_143_bias_m_read_readvariableop6
2savev2_adam_dense_308_kernel_m_read_readvariableop4
0savev2_adam_dense_308_bias_m_read_readvariableop6
2savev2_adam_dense_309_kernel_m_read_readvariableop4
0savev2_adam_dense_309_bias_m_read_readvariableop6
2savev2_adam_dense_310_kernel_m_read_readvariableop4
0savev2_adam_dense_310_bias_m_read_readvariableop7
3savev2_adam_conv2d_143_kernel_v_read_readvariableop5
1savev2_adam_conv2d_143_bias_v_read_readvariableop6
2savev2_adam_dense_308_kernel_v_read_readvariableop4
0savev2_adam_dense_308_bias_v_read_readvariableop6
2savev2_adam_dense_309_kernel_v_read_readvariableop4
0savev2_adam_dense_309_bias_v_read_readvariableop6
2savev2_adam_dense_310_kernel_v_read_readvariableop4
0savev2_adam_dense_310_bias_v_read_readvariableop
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
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*?
value?B?$B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB:layer-0/layer-0/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUEB:layer-0/layer-1/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*[
valueRBP$B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0,savev2_conv2d_143_kernel_read_readvariableop*savev2_conv2d_143_bias_read_readvariableop+savev2_dense_308_kernel_read_readvariableop)savev2_dense_308_bias_read_readvariableop+savev2_dense_309_kernel_read_readvariableop)savev2_dense_309_bias_read_readvariableop+savev2_dense_310_kernel_read_readvariableop)savev2_dense_310_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop#savev2_variable_read_readvariableop%savev2_variable_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop3savev2_adam_conv2d_143_kernel_m_read_readvariableop1savev2_adam_conv2d_143_bias_m_read_readvariableop2savev2_adam_dense_308_kernel_m_read_readvariableop0savev2_adam_dense_308_bias_m_read_readvariableop2savev2_adam_dense_309_kernel_m_read_readvariableop0savev2_adam_dense_309_bias_m_read_readvariableop2savev2_adam_dense_310_kernel_m_read_readvariableop0savev2_adam_dense_310_bias_m_read_readvariableop3savev2_adam_conv2d_143_kernel_v_read_readvariableop1savev2_adam_conv2d_143_bias_v_read_readvariableop2savev2_adam_dense_308_kernel_v_read_readvariableop0savev2_adam_dense_308_bias_v_read_readvariableop2savev2_adam_dense_309_kernel_v_read_readvariableop0savev2_adam_dense_309_bias_v_read_readvariableop2savev2_adam_dense_310_kernel_v_read_readvariableop0savev2_adam_dense_310_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *2
dtypes(
&2$			2
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
: : : : : ::: : : : : : :
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
: : 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
: :&"
 
_output_shapes
:
?*?:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?
: 

_output_shapes
:
:,(
&
_output_shapes
: : 

_output_shapes
: :&"
 
_output_shapes
:
?*?:!

_output_shapes	
:?:& "
 
_output_shapes
:
??:!!

_output_shapes	
:?:%"!

_output_shapes
:	?
: #

_output_shapes
:
:$

_output_shapes
: 
?	
?
F__inference_dense_309_layer_call_and_return_conditional_losses_2512012

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
?
g
H__inference_dropout_110_layer_call_and_return_conditional_losses_2512665

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
?
c
G__inference_flatten_99_layer_call_and_return_conditional_losses_2512686

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
?
?
<__inference_Training_Data_Augmentation_layer_call_fn_2511862
random_flip_input
unknown
identity??StatefulPartitionedCall?
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
GPU 2J 8? *`
f[RY
W__inference_Training_Data_Augmentation_layer_call_and_return_conditional_losses_25118572
StatefulPartitionedCall?
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
?$
?
I__inference_Fourth_Model_layer_call_and_return_conditional_losses_2512168

inputs
conv2d_143_2512144
conv2d_143_2512146
dense_308_2512152
dense_308_2512154
dense_309_2512157
dense_309_2512159
dense_310_2512162
dense_310_2512164
identity??"conv2d_143/StatefulPartitionedCall?!dense_308/StatefulPartitionedCall?!dense_309/StatefulPartitionedCall?!dense_310/StatefulPartitionedCall?
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
GPU 2J 8? *`
f[RY
W__inference_Training_Data_Augmentation_layer_call_and_return_conditional_losses_25118682,
*Training_Data_Augmentation/PartitionedCall?
"conv2d_143/StatefulPartitionedCallStatefulPartitionedCall3Training_Data_Augmentation/PartitionedCall:output:0conv2d_143_2512144conv2d_143_2512146*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_conv2d_143_layer_call_and_return_conditional_losses_25119132$
"conv2d_143/StatefulPartitionedCall?
!max_pooling2d_132/PartitionedCallPartitionedCall+conv2d_143/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_max_pooling2d_132_layer_call_and_return_conditional_losses_25118772#
!max_pooling2d_132/PartitionedCall?
dropout_110/PartitionedCallPartitionedCall*max_pooling2d_132/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_dropout_110_layer_call_and_return_conditional_losses_25119472
dropout_110/PartitionedCall?
flatten_99/PartitionedCallPartitionedCall$dropout_110/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????** 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_flatten_99_layer_call_and_return_conditional_losses_25119662
flatten_99/PartitionedCall?
!dense_308/StatefulPartitionedCallStatefulPartitionedCall#flatten_99/PartitionedCall:output:0dense_308_2512152dense_308_2512154*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_308_layer_call_and_return_conditional_losses_25119852#
!dense_308/StatefulPartitionedCall?
!dense_309/StatefulPartitionedCallStatefulPartitionedCall*dense_308/StatefulPartitionedCall:output:0dense_309_2512157dense_309_2512159*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dense_309_layer_call_and_return_conditional_losses_25120122#
!dense_309/StatefulPartitionedCall?
!dense_310/StatefulPartitionedCallStatefulPartitionedCall*dense_309/StatefulPartitionedCall:output:0dense_310_2512162dense_310_2512164*
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
GPU 2J 8? *O
fJRH
F__inference_dense_310_layer_call_and_return_conditional_losses_25120392#
!dense_310/StatefulPartitionedCall?
IdentityIdentity*dense_310/StatefulPartitionedCall:output:0#^conv2d_143/StatefulPartitionedCall"^dense_308/StatefulPartitionedCall"^dense_309/StatefulPartitionedCall"^dense_310/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::2H
"conv2d_143/StatefulPartitionedCall"conv2d_143/StatefulPartitionedCall2F
!dense_308/StatefulPartitionedCall!dense_308/StatefulPartitionedCall2F
!dense_309/StatefulPartitionedCall!dense_309/StatefulPartitionedCall2F
!dense_310/StatefulPartitionedCall!dense_310/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
<__inference_Training_Data_Augmentation_layer_call_fn_2512628

inputs
unknown
identity??StatefulPartitionedCall?
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
GPU 2J 8? *`
f[RY
W__inference_Training_Data_Augmentation_layer_call_and_return_conditional_losses_25118572
StatefulPartitionedCall?
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
?

?
G__inference_conv2d_143_layer_call_and_return_conditional_losses_2512644

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
~
W__inference_Training_Data_Augmentation_layer_call_and_return_conditional_losses_2511714
random_flip_input
identitym
IdentityIdentityrandom_flip_input*
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
_user_specified_namerandom_flip_input"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
u
 Training_Data_Augmentation_inputQ
2serving_default_Training_Data_Augmentation_input:0?????????=
	dense_3100
StatefulPartitionedCall:0?????????
tensorflow/serving/predict:??
?E
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer-4
layer_with_weights-1
layer-5
layer_with_weights-2
layer-6
layer_with_weights-3
layer-7
		optimizer

	variables
regularization_losses
trainable_variables
	keras_api

signatures
?_default_save_signature
+?&call_and_return_all_conditional_losses
?__call__"?B
_tf_keras_sequential?B{"class_name": "Sequential", "name": "Fourth_Model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "Fourth_Model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Training_Data_Augmentation_input"}}, {"class_name": "Sequential", "config": {"name": "Training_Data_Augmentation", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "random_flip_input"}}, {"class_name": "RandomFlip", "config": {"name": "random_flip", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "mode": "horizontal", "seed": null}}, {"class_name": "RandomRotation", "config": {"name": "random_rotation", "trainable": true, "dtype": "float32", "factor": 0.2, "fill_mode": "reflect", "fill_value": 0.0, "interpolation": "bilinear", "seed": null}}, {"class_name": "RandomContrast", "config": {"name": "random_contrast", "trainable": true, "dtype": "float32", "factor": 0.1, "seed": null}}]}}, {"class_name": "Conv2D", "config": {"name": "conv2d_143", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_132", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_110", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten_99", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_308", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_309", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_310", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "Fourth_Model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Training_Data_Augmentation_input"}}, {"class_name": "Sequential", "config": {"name": "Training_Data_Augmentation", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "random_flip_input"}}, {"class_name": "RandomFlip", "config": {"name": "random_flip", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "mode": "horizontal", "seed": null}}, {"class_name": "RandomRotation", "config": {"name": "random_rotation", "trainable": true, "dtype": "float32", "factor": 0.2, "fill_mode": "reflect", "fill_value": 0.0, "interpolation": "bilinear", "seed": null}}, {"class_name": "RandomContrast", "config": {"name": "random_contrast", "trainable": true, "dtype": "float32", "factor": 0.1, "seed": null}}]}}, {"class_name": "Conv2D", "config": {"name": "conv2d_143", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_132", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_110", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten_99", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_308", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_309", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_310", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "sparse_categorical_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.009999999776482582, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?
layer-0
layer-1
layer-2
	variables
regularization_losses
trainable_variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_sequential?{"class_name": "Sequential", "name": "Training_Data_Augmentation", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "Training_Data_Augmentation", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "random_flip_input"}}, {"class_name": "RandomFlip", "config": {"name": "random_flip", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "mode": "horizontal", "seed": null}}, {"class_name": "RandomRotation", "config": {"name": "random_rotation", "trainable": true, "dtype": "float32", "factor": 0.2, "fill_mode": "reflect", "fill_value": 0.0, "interpolation": "bilinear", "seed": null}}, {"class_name": "RandomContrast", "config": {"name": "random_contrast", "trainable": true, "dtype": "float32", "factor": 0.1, "seed": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "Training_Data_Augmentation", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "random_flip_input"}}, {"class_name": "RandomFlip", "config": {"name": "random_flip", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "mode": "horizontal", "seed": null}}, {"class_name": "RandomRotation", "config": {"name": "random_rotation", "trainable": true, "dtype": "float32", "factor": 0.2, "fill_mode": "reflect", "fill_value": 0.0, "interpolation": "bilinear", "seed": null}}, {"class_name": "RandomContrast", "config": {"name": "random_contrast", "trainable": true, "dtype": "float32", "factor": 0.1, "seed": null}}]}}}
?


kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"class_name": "Conv2D", "name": "conv2d_143", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_143", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 1]}}
?
	variables
regularization_losses
trainable_variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_132", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_132", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
 	variables
!regularization_losses
"trainable_variables
#	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_110", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_110", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
?
$	variables
%regularization_losses
&trainable_variables
'	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_99", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_99", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?

(kernel
)bias
*	variables
+regularization_losses
,trainable_variables
-	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_308", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_308", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 5408}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 5408]}}
?

.kernel
/bias
0	variables
1regularization_losses
2trainable_variables
3	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_309", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_309", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
?

4kernel
5bias
6	variables
7regularization_losses
8trainable_variables
9	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_310", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_310", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
?
:iter

;beta_1

<beta_2
	=decay
>learning_ratem~m(m?)m?.m?/m?4m?5m?v?v?(v?)v?.v?/v?4v?5v?"
	optimizer
X
0
1
(2
)3
.4
/5
46
57"
trackable_list_wrapper
 "
trackable_list_wrapper
X
0
1
(2
)3
.4
/5
46
57"
trackable_list_wrapper
?
?layer_metrics

	variables
regularization_losses

@layers
Alayer_regularization_losses
Bnon_trainable_variables
Cmetrics
trainable_variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
?
D_rng
E	keras_api"?
_tf_keras_layer?{"class_name": "RandomFlip", "name": "random_flip", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "stateful": false, "must_restore_from_config": true, "config": {"name": "random_flip", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "mode": "horizontal", "seed": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
F_rng
G	keras_api"?
_tf_keras_layer?{"class_name": "RandomRotation", "name": "random_rotation", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "random_rotation", "trainable": true, "dtype": "float32", "factor": 0.2, "fill_mode": "reflect", "fill_value": 0.0, "interpolation": "bilinear", "seed": null}}
?
H	keras_api"?
_tf_keras_layer?{"class_name": "RandomContrast", "name": "random_contrast", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "random_contrast", "trainable": true, "dtype": "float32", "factor": 0.1, "seed": null}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Ilayer_metrics
	variables
regularization_losses

Jlayers
Klayer_regularization_losses
Lnon_trainable_variables
Mmetrics
trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
+:) 2conv2d_143/kernel
: 2conv2d_143/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
Nlayer_metrics
	variables
regularization_losses

Olayers
Player_regularization_losses
Qnon_trainable_variables
Rmetrics
trainable_variables
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
Slayer_metrics
	variables
regularization_losses

Tlayers
Ulayer_regularization_losses
Vnon_trainable_variables
Wmetrics
trainable_variables
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
Xlayer_metrics
 	variables
!regularization_losses

Ylayers
Zlayer_regularization_losses
[non_trainable_variables
\metrics
"trainable_variables
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
]layer_metrics
$	variables
%regularization_losses

^layers
_layer_regularization_losses
`non_trainable_variables
ametrics
&trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
$:"
?*?2dense_308/kernel
:?2dense_308/bias
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
?
blayer_metrics
*	variables
+regularization_losses

clayers
dlayer_regularization_losses
enon_trainable_variables
fmetrics
,trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
$:"
??2dense_309/kernel
:?2dense_309/bias
.
.0
/1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
?
glayer_metrics
0	variables
1regularization_losses

hlayers
ilayer_regularization_losses
jnon_trainable_variables
kmetrics
2trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
#:!	?
2dense_310/kernel
:
2dense_310/bias
.
40
51"
trackable_list_wrapper
 "
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
?
llayer_metrics
6	variables
7regularization_losses

mlayers
nlayer_regularization_losses
onon_trainable_variables
pmetrics
8trainable_variables
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
 "
trackable_dict_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
q0
r1"
trackable_list_wrapper
.
s
_state_var"
_generic_user_object
"
_generic_user_object
.
t
_state_var"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
 "
trackable_dict_wrapper
5
0
1
2"
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
?
	utotal
	vcount
w	variables
x	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
?
	ytotal
	zcount
{
_fn_kwargs
|	variables
}	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}}
:	2Variable
:	2Variable
:  (2total
:  (2count
.
u0
v1"
trackable_list_wrapper
-
w	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
y0
z1"
trackable_list_wrapper
-
|	variables"
_generic_user_object
0:. 2Adam/conv2d_143/kernel/m
":  2Adam/conv2d_143/bias/m
):'
?*?2Adam/dense_308/kernel/m
": ?2Adam/dense_308/bias/m
):'
??2Adam/dense_309/kernel/m
": ?2Adam/dense_309/bias/m
(:&	?
2Adam/dense_310/kernel/m
!:
2Adam/dense_310/bias/m
0:. 2Adam/conv2d_143/kernel/v
":  2Adam/conv2d_143/bias/v
):'
?*?2Adam/dense_308/kernel/v
": ?2Adam/dense_308/bias/v
):'
??2Adam/dense_309/kernel/v
": ?2Adam/dense_309/bias/v
(:&	?
2Adam/dense_310/kernel/v
!:
2Adam/dense_310/bias/v
?2?
"__inference__wrapped_model_2511569?
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
annotations? *G?D
B??
 Training_Data_Augmentation_input?????????
?2?
I__inference_Fourth_Model_layer_call_and_return_conditional_losses_2512397
I__inference_Fourth_Model_layer_call_and_return_conditional_losses_2512433
I__inference_Fourth_Model_layer_call_and_return_conditional_losses_2512056
I__inference_Fourth_Model_layer_call_and_return_conditional_losses_2512084?
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
.__inference_Fourth_Model_layer_call_fn_2512456
.__inference_Fourth_Model_layer_call_fn_2512477
.__inference_Fourth_Model_layer_call_fn_2512187
.__inference_Fourth_Model_layer_call_fn_2512138?
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
?2?
W__inference_Training_Data_Augmentation_layer_call_and_return_conditional_losses_2511714
W__inference_Training_Data_Augmentation_layer_call_and_return_conditional_losses_2512617
W__inference_Training_Data_Augmentation_layer_call_and_return_conditional_losses_2512621
W__inference_Training_Data_Augmentation_layer_call_and_return_conditional_losses_2511710?
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
<__inference_Training_Data_Augmentation_layer_call_fn_2512628
<__inference_Training_Data_Augmentation_layer_call_fn_2511862
<__inference_Training_Data_Augmentation_layer_call_fn_2512633
<__inference_Training_Data_Augmentation_layer_call_fn_2511871?
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
G__inference_conv2d_143_layer_call_and_return_conditional_losses_2512644?
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
,__inference_conv2d_143_layer_call_fn_2512653?
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
N__inference_max_pooling2d_132_layer_call_and_return_conditional_losses_2511877?
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
3__inference_max_pooling2d_132_layer_call_fn_2511883?
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
H__inference_dropout_110_layer_call_and_return_conditional_losses_2512665
H__inference_dropout_110_layer_call_and_return_conditional_losses_2512670?
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
-__inference_dropout_110_layer_call_fn_2512680
-__inference_dropout_110_layer_call_fn_2512675?
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
G__inference_flatten_99_layer_call_and_return_conditional_losses_2512686?
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
,__inference_flatten_99_layer_call_fn_2512691?
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
F__inference_dense_308_layer_call_and_return_conditional_losses_2512702?
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
+__inference_dense_308_layer_call_fn_2512711?
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
F__inference_dense_309_layer_call_and_return_conditional_losses_2512722?
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
+__inference_dense_309_layer_call_fn_2512731?
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
F__inference_dense_310_layer_call_and_return_conditional_losses_2512742?
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
+__inference_dense_310_layer_call_fn_2512751?
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
%__inference_signature_wrapper_2512218 Training_Data_Augmentation_input"?
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
I__inference_Fourth_Model_layer_call_and_return_conditional_losses_2512056?	t()./45Y?V
O?L
B??
 Training_Data_Augmentation_input?????????
p

 
? "%?"
?
0?????????

? ?
I__inference_Fourth_Model_layer_call_and_return_conditional_losses_2512084?()./45Y?V
O?L
B??
 Training_Data_Augmentation_input?????????
p 

 
? "%?"
?
0?????????

? ?
I__inference_Fourth_Model_layer_call_and_return_conditional_losses_2512397s	t()./45??<
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
I__inference_Fourth_Model_layer_call_and_return_conditional_losses_2512433r()./45??<
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
.__inference_Fourth_Model_layer_call_fn_2512138?	t()./45Y?V
O?L
B??
 Training_Data_Augmentation_input?????????
p

 
? "??????????
?
.__inference_Fourth_Model_layer_call_fn_2512187()./45Y?V
O?L
B??
 Training_Data_Augmentation_input?????????
p 

 
? "??????????
?
.__inference_Fourth_Model_layer_call_fn_2512456f	t()./45??<
5?2
(?%
inputs?????????
p

 
? "??????????
?
.__inference_Fourth_Model_layer_call_fn_2512477e()./45??<
5?2
(?%
inputs?????????
p 

 
? "??????????
?
W__inference_Training_Data_Augmentation_layer_call_and_return_conditional_losses_2511710~tJ?G
@?=
3?0
random_flip_input?????????
p

 
? "-?*
#? 
0?????????
? ?
W__inference_Training_Data_Augmentation_layer_call_and_return_conditional_losses_2511714{J?G
@?=
3?0
random_flip_input?????????
p 

 
? "-?*
#? 
0?????????
? ?
W__inference_Training_Data_Augmentation_layer_call_and_return_conditional_losses_2512617st??<
5?2
(?%
inputs?????????
p

 
? "-?*
#? 
0?????????
? ?
W__inference_Training_Data_Augmentation_layer_call_and_return_conditional_losses_2512621p??<
5?2
(?%
inputs?????????
p 

 
? "-?*
#? 
0?????????
? ?
<__inference_Training_Data_Augmentation_layer_call_fn_2511862qtJ?G
@?=
3?0
random_flip_input?????????
p

 
? " ???????????
<__inference_Training_Data_Augmentation_layer_call_fn_2511871nJ?G
@?=
3?0
random_flip_input?????????
p 

 
? " ???????????
<__inference_Training_Data_Augmentation_layer_call_fn_2512628ft??<
5?2
(?%
inputs?????????
p

 
? " ???????????
<__inference_Training_Data_Augmentation_layer_call_fn_2512633c??<
5?2
(?%
inputs?????????
p 

 
? " ???????????
"__inference__wrapped_model_2511569?()./45Q?N
G?D
B??
 Training_Data_Augmentation_input?????????
? "5?2
0
	dense_310#? 
	dense_310?????????
?
G__inference_conv2d_143_layer_call_and_return_conditional_losses_2512644l7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0????????? 
? ?
,__inference_conv2d_143_layer_call_fn_2512653_7?4
-?*
(?%
inputs?????????
? " ?????????? ?
F__inference_dense_308_layer_call_and_return_conditional_losses_2512702^()0?-
&?#
!?
inputs??????????*
? "&?#
?
0??????????
? ?
+__inference_dense_308_layer_call_fn_2512711Q()0?-
&?#
!?
inputs??????????*
? "????????????
F__inference_dense_309_layer_call_and_return_conditional_losses_2512722^./0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
+__inference_dense_309_layer_call_fn_2512731Q./0?-
&?#
!?
inputs??????????
? "????????????
F__inference_dense_310_layer_call_and_return_conditional_losses_2512742]450?-
&?#
!?
inputs??????????
? "%?"
?
0?????????

? 
+__inference_dense_310_layer_call_fn_2512751P450?-
&?#
!?
inputs??????????
? "??????????
?
H__inference_dropout_110_layer_call_and_return_conditional_losses_2512665l;?8
1?.
(?%
inputs????????? 
p
? "-?*
#? 
0????????? 
? ?
H__inference_dropout_110_layer_call_and_return_conditional_losses_2512670l;?8
1?.
(?%
inputs????????? 
p 
? "-?*
#? 
0????????? 
? ?
-__inference_dropout_110_layer_call_fn_2512675_;?8
1?.
(?%
inputs????????? 
p
? " ?????????? ?
-__inference_dropout_110_layer_call_fn_2512680_;?8
1?.
(?%
inputs????????? 
p 
? " ?????????? ?
G__inference_flatten_99_layer_call_and_return_conditional_losses_2512686a7?4
-?*
(?%
inputs????????? 
? "&?#
?
0??????????*
? ?
,__inference_flatten_99_layer_call_fn_2512691T7?4
-?*
(?%
inputs????????? 
? "???????????*?
N__inference_max_pooling2d_132_layer_call_and_return_conditional_losses_2511877?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
3__inference_max_pooling2d_132_layer_call_fn_2511883?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
%__inference_signature_wrapper_2512218?()./45u?r
? 
k?h
f
 Training_Data_Augmentation_inputB??
 Training_Data_Augmentation_input?????????"5?2
0
	dense_310#? 
	dense_310?????????
