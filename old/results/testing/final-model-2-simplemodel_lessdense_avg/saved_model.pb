ó
¿£
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
¾
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
 "serve*2.3.02unknown8

conv2d_101/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameconv2d_101/kernel

%conv2d_101/kernel/Read/ReadVariableOpReadVariableOpconv2d_101/kernel*&
_output_shapes
:@*
dtype0
v
conv2d_101/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_nameconv2d_101/bias
o
#conv2d_101/bias/Read/ReadVariableOpReadVariableOpconv2d_101/bias*
_output_shapes
:@*
dtype0
~
dense_201/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ÀT*!
shared_namedense_201/kernel
w
$dense_201/kernel/Read/ReadVariableOpReadVariableOpdense_201/kernel* 
_output_shapes
:
ÀT*
dtype0
u
dense_201/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_201/bias
n
"dense_201/bias/Read/ReadVariableOpReadVariableOpdense_201/bias*
_output_shapes	
:*
dtype0
}
dense_202/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*!
shared_namedense_202/kernel
v
$dense_202/kernel/Read/ReadVariableOpReadVariableOpdense_202/kernel*
_output_shapes
:	@*
dtype0
t
dense_202/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_202/bias
m
"dense_202/bias/Read/ReadVariableOpReadVariableOpdense_202/bias*
_output_shapes
:@*
dtype0
|
dense_203/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@
*!
shared_namedense_203/kernel
u
$dense_203/kernel/Read/ReadVariableOpReadVariableOpdense_203/kernel*
_output_shapes

:@
*
dtype0
t
dense_203/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_203/bias
m
"dense_203/bias/Read/ReadVariableOpReadVariableOpdense_203/bias*
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

Adam/conv2d_101/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_nameAdam/conv2d_101/kernel/m

,Adam/conv2d_101/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_101/kernel/m*&
_output_shapes
:@*
dtype0

Adam/conv2d_101/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdam/conv2d_101/bias/m
}
*Adam/conv2d_101/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_101/bias/m*
_output_shapes
:@*
dtype0

Adam/dense_201/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ÀT*(
shared_nameAdam/dense_201/kernel/m

+Adam/dense_201/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_201/kernel/m* 
_output_shapes
:
ÀT*
dtype0

Adam/dense_201/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_201/bias/m
|
)Adam/dense_201/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_201/bias/m*
_output_shapes	
:*
dtype0

Adam/dense_202/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*(
shared_nameAdam/dense_202/kernel/m

+Adam/dense_202/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_202/kernel/m*
_output_shapes
:	@*
dtype0

Adam/dense_202/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/dense_202/bias/m
{
)Adam/dense_202/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_202/bias/m*
_output_shapes
:@*
dtype0

Adam/dense_203/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@
*(
shared_nameAdam/dense_203/kernel/m

+Adam/dense_203/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_203/kernel/m*
_output_shapes

:@
*
dtype0

Adam/dense_203/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/dense_203/bias/m
{
)Adam/dense_203/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_203/bias/m*
_output_shapes
:
*
dtype0

Adam/conv2d_101/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_nameAdam/conv2d_101/kernel/v

,Adam/conv2d_101/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_101/kernel/v*&
_output_shapes
:@*
dtype0

Adam/conv2d_101/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdam/conv2d_101/bias/v
}
*Adam/conv2d_101/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_101/bias/v*
_output_shapes
:@*
dtype0

Adam/dense_201/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ÀT*(
shared_nameAdam/dense_201/kernel/v

+Adam/dense_201/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_201/kernel/v* 
_output_shapes
:
ÀT*
dtype0

Adam/dense_201/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_201/bias/v
|
)Adam/dense_201/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_201/bias/v*
_output_shapes	
:*
dtype0

Adam/dense_202/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*(
shared_nameAdam/dense_202/kernel/v

+Adam/dense_202/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_202/kernel/v*
_output_shapes
:	@*
dtype0

Adam/dense_202/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/dense_202/bias/v
{
)Adam/dense_202/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_202/bias/v*
_output_shapes
:@*
dtype0

Adam/dense_203/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@
*(
shared_nameAdam/dense_203/kernel/v

+Adam/dense_203/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_203/kernel/v*
_output_shapes

:@
*
dtype0

Adam/dense_203/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/dense_203/bias/v
{
)Adam/dense_203/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_203/bias/v*
_output_shapes
:
*
dtype0

NoOpNoOp
Ð@
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*@
value@Bþ? B÷?
Á
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

regularization_losses
	variables
trainable_variables
	keras_api

signatures
y
layer-0
layer-1
layer-2
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
R
regularization_losses
	variables
trainable_variables
	keras_api
R
 regularization_losses
!	variables
"trainable_variables
#	keras_api
R
$regularization_losses
%	variables
&trainable_variables
'	keras_api
h

(kernel
)bias
*regularization_losses
+	variables
,trainable_variables
-	keras_api
h

.kernel
/bias
0regularization_losses
1	variables
2trainable_variables
3	keras_api
h

4kernel
5bias
6regularization_losses
7	variables
8trainable_variables
9	keras_api
à
:iter

;beta_1

<beta_2
	=decay
>learning_ratemm(m)m.m/m4m5mvv(v )v¡.v¢/v£4v¤5v¥
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
8
0
1
(2
)3
.4
/5
46
57
­

?layers
@layer_metrics
Anon_trainable_variables

regularization_losses
	variables
trainable_variables
Bmetrics
Clayer_regularization_losses
 
\
D_rng
Eregularization_losses
F	variables
Gtrainable_variables
H	keras_api
\
I_rng
Jregularization_losses
K	variables
Ltrainable_variables
M	keras_api
R
Nregularization_losses
O	variables
Ptrainable_variables
Q	keras_api
 
 
 
­

Rlayers
Slayer_metrics
Tnon_trainable_variables
regularization_losses
	variables
trainable_variables
Umetrics
Vlayer_regularization_losses
][
VARIABLE_VALUEconv2d_101/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEconv2d_101/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
­

Wlayers
Xlayer_metrics
Ynon_trainable_variables
regularization_losses
	variables
trainable_variables
Zmetrics
[layer_regularization_losses
 
 
 
­

\layers
]layer_metrics
^non_trainable_variables
regularization_losses
	variables
trainable_variables
_metrics
`layer_regularization_losses
 
 
 
­

alayers
blayer_metrics
cnon_trainable_variables
 regularization_losses
!	variables
"trainable_variables
dmetrics
elayer_regularization_losses
 
 
 
­

flayers
glayer_metrics
hnon_trainable_variables
$regularization_losses
%	variables
&trainable_variables
imetrics
jlayer_regularization_losses
\Z
VARIABLE_VALUEdense_201/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_201/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

(0
)1

(0
)1
­

klayers
llayer_metrics
mnon_trainable_variables
*regularization_losses
+	variables
,trainable_variables
nmetrics
olayer_regularization_losses
\Z
VARIABLE_VALUEdense_202/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_202/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

.0
/1

.0
/1
­

players
qlayer_metrics
rnon_trainable_variables
0regularization_losses
1	variables
2trainable_variables
smetrics
tlayer_regularization_losses
\Z
VARIABLE_VALUEdense_203/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_203/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

40
51

40
51
­

ulayers
vlayer_metrics
wnon_trainable_variables
6regularization_losses
7	variables
8trainable_variables
xmetrics
ylayer_regularization_losses
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
z0
{1
 

|
_state_var
 
 
 
¯

}layers
~layer_metrics
non_trainable_variables
Eregularization_losses
F	variables
Gtrainable_variables
metrics
 layer_regularization_losses


_state_var
 
 
 
²
layers
layer_metrics
non_trainable_variables
Jregularization_losses
K	variables
Ltrainable_variables
metrics
 layer_regularization_losses
 
 
 
²
layers
layer_metrics
non_trainable_variables
Nregularization_losses
O	variables
Ptrainable_variables
metrics
 layer_regularization_losses
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
 
8

total

count
	variables
	keras_api
I

total

count

_fn_kwargs
	variables
	keras_api
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
0
1

	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

	variables
~
VARIABLE_VALUEAdam/conv2d_101/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_101/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_201/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_201/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_202/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_202/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_203/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_203/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/conv2d_101/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/conv2d_101/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_201/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_201/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_202/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_202/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_203/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_203/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
£
0serving_default_Training_Data_Augmentation_inputPlaceholder*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*$
shape:ÿÿÿÿÿÿÿÿÿ
æ
StatefulPartitionedCallStatefulPartitionedCall0serving_default_Training_Data_Augmentation_inputconv2d_101/kernelconv2d_101/biasdense_201/kerneldense_201/biasdense_202/kerneldense_202/biasdense_203/kerneldense_203/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *.
f)R'
%__inference_signature_wrapper_1737210
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename%conv2d_101/kernel/Read/ReadVariableOp#conv2d_101/bias/Read/ReadVariableOp$dense_201/kernel/Read/ReadVariableOp"dense_201/bias/Read/ReadVariableOp$dense_202/kernel/Read/ReadVariableOp"dense_202/bias/Read/ReadVariableOp$dense_203/kernel/Read/ReadVariableOp"dense_203/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOpVariable/Read/ReadVariableOpVariable_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp,Adam/conv2d_101/kernel/m/Read/ReadVariableOp*Adam/conv2d_101/bias/m/Read/ReadVariableOp+Adam/dense_201/kernel/m/Read/ReadVariableOp)Adam/dense_201/bias/m/Read/ReadVariableOp+Adam/dense_202/kernel/m/Read/ReadVariableOp)Adam/dense_202/bias/m/Read/ReadVariableOp+Adam/dense_203/kernel/m/Read/ReadVariableOp)Adam/dense_203/bias/m/Read/ReadVariableOp,Adam/conv2d_101/kernel/v/Read/ReadVariableOp*Adam/conv2d_101/bias/v/Read/ReadVariableOp+Adam/dense_201/kernel/v/Read/ReadVariableOp)Adam/dense_201/bias/v/Read/ReadVariableOp+Adam/dense_202/kernel/v/Read/ReadVariableOp)Adam/dense_202/bias/v/Read/ReadVariableOp+Adam/dense_203/kernel/v/Read/ReadVariableOp)Adam/dense_203/bias/v/Read/ReadVariableOpConst*0
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
GPU 2J 8 *)
f$R"
 __inference__traced_save_1738136
Ã
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_101/kernelconv2d_101/biasdense_201/kerneldense_201/biasdense_202/kerneldense_202/biasdense_203/kerneldense_203/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateVariable
Variable_1totalcounttotal_1count_1Adam/conv2d_101/kernel/mAdam/conv2d_101/bias/mAdam/dense_201/kernel/mAdam/dense_201/bias/mAdam/dense_202/kernel/mAdam/dense_202/bias/mAdam/dense_203/kernel/mAdam/dense_203/bias/mAdam/conv2d_101/kernel/vAdam/conv2d_101/bias/vAdam/dense_201/kernel/vAdam/dense_201/bias/vAdam/dense_202/kernel/vAdam/dense_202/bias/vAdam/dense_203/kernel/vAdam/dense_203/bias/v*/
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
GPU 2J 8 *,
f'R%
#__inference__traced_restore_1738251Ìå
ò
d
H__inference_random_flip_layer_call_and_return_conditional_losses_1736657

inputs
identityb
IdentityIdentityinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ä

W__inference_Training_Data_Augmentation_layer_call_and_return_conditional_losses_1736846

inputs
random_rotation_1736841
identity¢'random_contrast/StatefulPartitionedCall¢#random_flip/StatefulPartitionedCall¢'random_rotation/StatefulPartitionedCallþ
#random_flip/StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_random_flip_layer_call_and_return_conditional_losses_17366532%
#random_flip/StatefulPartitionedCallÊ
'random_rotation/StatefulPartitionedCallStatefulPartitionedCall,random_flip/StatefulPartitionedCall:output:0random_rotation_1736841*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_random_rotation_layer_call_and_return_conditional_losses_17367752)
'random_rotation/StatefulPartitionedCallÚ
'random_contrast/StatefulPartitionedCallStatefulPartitionedCall0random_rotation/StatefulPartitionedCall:output:0$^random_flip/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_random_contrast_layer_call_and_return_conditional_losses_17368092)
'random_contrast/StatefulPartitionedCall
IdentityIdentity0random_contrast/StatefulPartitionedCall:output:0(^random_contrast/StatefulPartitionedCall$^random_flip/StatefulPartitionedCall(^random_rotation/StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ:2R
'random_contrast/StatefulPartitionedCall'random_contrast/StatefulPartitionedCall2J
#random_flip/StatefulPartitionedCall#random_flip/StatefulPartitionedCall2R
'random_rotation/StatefulPartitionedCall'random_rotation/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Æ
g
H__inference_dropout_101_layer_call_and_return_conditional_losses_1737655

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¼
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2
dropout/GreaterEqual/yÆ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ø
X
<__inference_Training_Data_Augmentation_layer_call_fn_1737623

inputs
identityÝ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *`
f[RY
W__inference_Training_Data_Augmentation_layer_call_and_return_conditional_losses_17368602
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ã
d
H__inference_random_flip_layer_call_and_return_conditional_losses_1737839

inputs
identity}
IdentityIdentityinputs*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
(
g
H__inference_random_flip_layer_call_and_return_conditional_losses_1737769

inputs
identity¿
)random_flip_left_right/control_dependencyIdentityinputs*
T0*
_class
loc:@inputs*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)random_flip_left_right/control_dependency
random_flip_left_right/ShapeShape2random_flip_left_right/control_dependency:output:0*
T0*
_output_shapes
:2
random_flip_left_right/Shape¢
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
,random_flip_left_right/strided_slice/stack_2ì
$random_flip_left_right/strided_sliceStridedSlice%random_flip_left_right/Shape:output:03random_flip_left_right/strided_slice/stack:output:05random_flip_left_right/strided_slice/stack_1:output:05random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$random_flip_left_right/strided_slice¿
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
)random_flip_left_right/random_uniform/maxì
3random_flip_left_right/random_uniform/RandomUniformRandomUniform4random_flip_left_right/random_uniform/shape:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype025
3random_flip_left_right/random_uniform/RandomUniformý
)random_flip_left_right/random_uniform/MulMul<random_flip_left_right/random_uniform/RandomUniform:output:02random_flip_left_right/random_uniform/max:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
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
&random_flip_left_right/Reshape/shape/3Ä
$random_flip_left_right/Reshape/shapePack-random_flip_left_right/strided_slice:output:0/random_flip_left_right/Reshape/shape/1:output:0/random_flip_left_right/Reshape/shape/2:output:0/random_flip_left_right/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2&
$random_flip_left_right/Reshape/shapeã
random_flip_left_right/ReshapeReshape-random_flip_left_right/random_uniform/Mul:z:0-random_flip_left_right/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
random_flip_left_right/Reshape¨
random_flip_left_right/RoundRound'random_flip_left_right/Reshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
random_flip_left_right/Round
%random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:2'
%random_flip_left_right/ReverseV2/axisï
 random_flip_left_right/ReverseV2	ReverseV22random_flip_left_right/control_dependency:output:0.random_flip_left_right/ReverseV2/axis:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 random_flip_left_right/ReverseV2Æ
random_flip_left_right/mulMul random_flip_left_right/Round:y:0)random_flip_left_right/ReverseV2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
random_flip_left_right/mul
random_flip_left_right/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
random_flip_left_right/sub/xÂ
random_flip_left_right/subSub%random_flip_left_right/sub/x:output:0 random_flip_left_right/Round:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
random_flip_left_right/subÑ
random_flip_left_right/mul_1Mulrandom_flip_left_right/sub:z:02random_flip_left_right/control_dependency:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
random_flip_left_right/mul_1½
random_flip_left_right/addAddV2random_flip_left_right/mul:z:0 random_flip_left_right/mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
random_flip_left_right/addz
IdentityIdentityrandom_flip_left_right/add:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ò
d
H__inference_random_flip_layer_call_and_return_conditional_losses_1737773

inputs
identityb
IdentityIdentityinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
´
®
F__inference_dense_201_layer_call_and_return_conditional_losses_1736977

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ÀT*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÀT:::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀT
 
_user_specified_nameinputs
´
®
F__inference_dense_201_layer_call_and_return_conditional_losses_1737692

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ÀT*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÀT:::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀT
 
_user_specified_nameinputs
ë
è
W__inference_Training_Data_Augmentation_layer_call_and_return_conditional_losses_1737607

inputs=
9random_rotation_stateful_uniform_statefuluniform_resource
identity¢0random_rotation/stateful_uniform/StatefulUniform×
5random_flip/random_flip_left_right/control_dependencyIdentityinputs*
T0*
_class
loc:@inputs*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ27
5random_flip/random_flip_left_right/control_dependencyÂ
(random_flip/random_flip_left_right/ShapeShape>random_flip/random_flip_left_right/control_dependency:output:0*
T0*
_output_shapes
:2*
(random_flip/random_flip_left_right/Shapeº
6random_flip/random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 28
6random_flip/random_flip_left_right/strided_slice/stack¾
8random_flip/random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8random_flip/random_flip_left_right/strided_slice/stack_1¾
8random_flip/random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8random_flip/random_flip_left_right/strided_slice/stack_2´
0random_flip/random_flip_left_right/strided_sliceStridedSlice1random_flip/random_flip_left_right/Shape:output:0?random_flip/random_flip_left_right/strided_slice/stack:output:0Arandom_flip/random_flip_left_right/strided_slice/stack_1:output:0Arandom_flip/random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask22
0random_flip/random_flip_left_right/strided_sliceã
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
:ÿÿÿÿÿÿÿÿÿ*
dtype02A
?random_flip/random_flip_left_right/random_uniform/RandomUniform­
5random_flip/random_flip_left_right/random_uniform/MulMulHrandom_flip/random_flip_left_right/random_uniform/RandomUniform:output:0>random_flip/random_flip_left_right/random_uniform/max:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ27
5random_flip/random_flip_left_right/random_uniform/Mulª
2random_flip/random_flip_left_right/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :24
2random_flip/random_flip_left_right/Reshape/shape/1ª
2random_flip/random_flip_left_right/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :24
2random_flip/random_flip_left_right/Reshape/shape/2ª
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
:ÿÿÿÿÿÿÿÿÿ2,
*random_flip/random_flip_left_right/ReshapeÌ
(random_flip/random_flip_left_right/RoundRound3random_flip/random_flip_left_right/Reshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
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
:ÿÿÿÿÿÿÿÿÿ2.
,random_flip/random_flip_left_right/ReverseV2ö
&random_flip/random_flip_left_right/mulMul,random_flip/random_flip_left_right/Round:y:05random_flip/random_flip_left_right/ReverseV2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&random_flip/random_flip_left_right/mul
(random_flip/random_flip_left_right/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2*
(random_flip/random_flip_left_right/sub/xò
&random_flip/random_flip_left_right/subSub1random_flip/random_flip_left_right/sub/x:output:0,random_flip/random_flip_left_right/Round:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&random_flip/random_flip_left_right/sub
(random_flip/random_flip_left_right/mul_1Mul*random_flip/random_flip_left_right/sub:z:0>random_flip/random_flip_left_right/control_dependency:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(random_flip/random_flip_left_right/mul_1í
&random_flip/random_flip_left_right/addAddV2*random_flip/random_flip_left_right/mul:z:0,random_flip/random_flip_left_right/mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
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
%random_rotation/strided_slice/stack_2Â
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
'random_rotation/strided_slice_1/stack_2Ì
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
'random_rotation/strided_slice_2/stack_2Ì
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
random_rotation/Cast_1®
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
 *|Ù ¿2&
$random_rotation/stateful_uniform/min
$random_rotation/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *|Ù ?2&
$random_rotation/stateful_uniform/maxº
:random_rotation/stateful_uniform/StatefulUniform/algorithmConst*
_output_shapes
: *
dtype0	*
value	B	 R2<
:random_rotation/stateful_uniform/StatefulUniform/algorithmà
0random_rotation/stateful_uniform/StatefulUniformStatefulUniform9random_rotation_stateful_uniform_statefuluniform_resourceCrandom_rotation/stateful_uniform/StatefulUniform/algorithm:output:0/random_rotation/stateful_uniform/shape:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shape_dtype022
0random_rotation/stateful_uniform/StatefulUniformÒ
$random_rotation/stateful_uniform/subSub-random_rotation/stateful_uniform/max:output:0-random_rotation/stateful_uniform/min:output:0*
T0*
_output_shapes
: 2&
$random_rotation/stateful_uniform/subæ
$random_rotation/stateful_uniform/mulMul9random_rotation/stateful_uniform/StatefulUniform:output:0(random_rotation/stateful_uniform/sub:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$random_rotation/stateful_uniform/mulÒ
 random_rotation/stateful_uniformAdd(random_rotation/stateful_uniform/mul:z:0-random_rotation/stateful_uniform/min:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 random_rotation/stateful_uniform
%random_rotation/rotation_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2'
%random_rotation/rotation_matrix/sub/y¾
#random_rotation/rotation_matrix/subSubrandom_rotation/Cast_1:y:0.random_rotation/rotation_matrix/sub/y:output:0*
T0*
_output_shapes
: 2%
#random_rotation/rotation_matrix/sub¥
#random_rotation/rotation_matrix/CosCos$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#random_rotation/rotation_matrix/Cos
'random_rotation/rotation_matrix/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2)
'random_rotation/rotation_matrix/sub_1/yÄ
%random_rotation/rotation_matrix/sub_1Subrandom_rotation/Cast_1:y:00random_rotation/rotation_matrix/sub_1/y:output:0*
T0*
_output_shapes
: 2'
%random_rotation/rotation_matrix/sub_1Ó
#random_rotation/rotation_matrix/mulMul'random_rotation/rotation_matrix/Cos:y:0)random_rotation/rotation_matrix/sub_1:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#random_rotation/rotation_matrix/mul¥
#random_rotation/rotation_matrix/SinSin$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#random_rotation/rotation_matrix/Sin
'random_rotation/rotation_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2)
'random_rotation/rotation_matrix/sub_2/yÂ
%random_rotation/rotation_matrix/sub_2Subrandom_rotation/Cast:y:00random_rotation/rotation_matrix/sub_2/y:output:0*
T0*
_output_shapes
: 2'
%random_rotation/rotation_matrix/sub_2×
%random_rotation/rotation_matrix/mul_1Mul'random_rotation/rotation_matrix/Sin:y:0)random_rotation/rotation_matrix/sub_2:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%random_rotation/rotation_matrix/mul_1×
%random_rotation/rotation_matrix/sub_3Sub'random_rotation/rotation_matrix/mul:z:0)random_rotation/rotation_matrix/mul_1:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%random_rotation/rotation_matrix/sub_3×
%random_rotation/rotation_matrix/sub_4Sub'random_rotation/rotation_matrix/sub:z:0)random_rotation/rotation_matrix/sub_3:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%random_rotation/rotation_matrix/sub_4
)random_rotation/rotation_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2+
)random_rotation/rotation_matrix/truediv/yê
'random_rotation/rotation_matrix/truedivRealDiv)random_rotation/rotation_matrix/sub_4:z:02random_rotation/rotation_matrix/truediv/y:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'random_rotation/rotation_matrix/truediv
'random_rotation/rotation_matrix/sub_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2)
'random_rotation/rotation_matrix/sub_5/yÂ
%random_rotation/rotation_matrix/sub_5Subrandom_rotation/Cast:y:00random_rotation/rotation_matrix/sub_5/y:output:0*
T0*
_output_shapes
: 2'
%random_rotation/rotation_matrix/sub_5©
%random_rotation/rotation_matrix/Sin_1Sin$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%random_rotation/rotation_matrix/Sin_1
'random_rotation/rotation_matrix/sub_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2)
'random_rotation/rotation_matrix/sub_6/yÄ
%random_rotation/rotation_matrix/sub_6Subrandom_rotation/Cast_1:y:00random_rotation/rotation_matrix/sub_6/y:output:0*
T0*
_output_shapes
: 2'
%random_rotation/rotation_matrix/sub_6Ù
%random_rotation/rotation_matrix/mul_2Mul)random_rotation/rotation_matrix/Sin_1:y:0)random_rotation/rotation_matrix/sub_6:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%random_rotation/rotation_matrix/mul_2©
%random_rotation/rotation_matrix/Cos_1Cos$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%random_rotation/rotation_matrix/Cos_1
'random_rotation/rotation_matrix/sub_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2)
'random_rotation/rotation_matrix/sub_7/yÂ
%random_rotation/rotation_matrix/sub_7Subrandom_rotation/Cast:y:00random_rotation/rotation_matrix/sub_7/y:output:0*
T0*
_output_shapes
: 2'
%random_rotation/rotation_matrix/sub_7Ù
%random_rotation/rotation_matrix/mul_3Mul)random_rotation/rotation_matrix/Cos_1:y:0)random_rotation/rotation_matrix/sub_7:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%random_rotation/rotation_matrix/mul_3×
#random_rotation/rotation_matrix/addAddV2)random_rotation/rotation_matrix/mul_2:z:0)random_rotation/rotation_matrix/mul_3:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#random_rotation/rotation_matrix/add×
%random_rotation/rotation_matrix/sub_8Sub)random_rotation/rotation_matrix/sub_5:z:0'random_rotation/rotation_matrix/add:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%random_rotation/rotation_matrix/sub_8
+random_rotation/rotation_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2-
+random_rotation/rotation_matrix/truediv_1/yð
)random_rotation/rotation_matrix/truediv_1RealDiv)random_rotation/rotation_matrix/sub_8:z:04random_rotation/rotation_matrix/truediv_1/y:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)random_rotation/rotation_matrix/truediv_1¢
%random_rotation/rotation_matrix/ShapeShape$random_rotation/stateful_uniform:z:0*
T0*
_output_shapes
:2'
%random_rotation/rotation_matrix/Shape´
3random_rotation/rotation_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3random_rotation/rotation_matrix/strided_slice/stack¸
5random_rotation/rotation_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5random_rotation/rotation_matrix/strided_slice/stack_1¸
5random_rotation/rotation_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5random_rotation/rotation_matrix/strided_slice/stack_2¢
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
:ÿÿÿÿÿÿÿÿÿ2'
%random_rotation/rotation_matrix/Cos_2¿
5random_rotation/rotation_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        27
5random_rotation/rotation_matrix/strided_slice_1/stackÃ
7random_rotation/rotation_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation/rotation_matrix/strided_slice_1/stack_1Ã
7random_rotation/rotation_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7random_rotation/rotation_matrix/strided_slice_1/stack_2×
/random_rotation/rotation_matrix/strided_slice_1StridedSlice)random_rotation/rotation_matrix/Cos_2:y:0>random_rotation/rotation_matrix/strided_slice_1/stack:output:0@random_rotation/rotation_matrix/strided_slice_1/stack_1:output:0@random_rotation/rotation_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask21
/random_rotation/rotation_matrix/strided_slice_1©
%random_rotation/rotation_matrix/Sin_2Sin$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%random_rotation/rotation_matrix/Sin_2¿
5random_rotation/rotation_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        27
5random_rotation/rotation_matrix/strided_slice_2/stackÃ
7random_rotation/rotation_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation/rotation_matrix/strided_slice_2/stack_1Ã
7random_rotation/rotation_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7random_rotation/rotation_matrix/strided_slice_2/stack_2×
/random_rotation/rotation_matrix/strided_slice_2StridedSlice)random_rotation/rotation_matrix/Sin_2:y:0>random_rotation/rotation_matrix/strided_slice_2/stack:output:0@random_rotation/rotation_matrix/strided_slice_2/stack_1:output:0@random_rotation/rotation_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask21
/random_rotation/rotation_matrix/strided_slice_2½
#random_rotation/rotation_matrix/NegNeg8random_rotation/rotation_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#random_rotation/rotation_matrix/Neg¿
5random_rotation/rotation_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        27
5random_rotation/rotation_matrix/strided_slice_3/stackÃ
7random_rotation/rotation_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation/rotation_matrix/strided_slice_3/stack_1Ã
7random_rotation/rotation_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7random_rotation/rotation_matrix/strided_slice_3/stack_2Ù
/random_rotation/rotation_matrix/strided_slice_3StridedSlice+random_rotation/rotation_matrix/truediv:z:0>random_rotation/rotation_matrix/strided_slice_3/stack:output:0@random_rotation/rotation_matrix/strided_slice_3/stack_1:output:0@random_rotation/rotation_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask21
/random_rotation/rotation_matrix/strided_slice_3©
%random_rotation/rotation_matrix/Sin_3Sin$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%random_rotation/rotation_matrix/Sin_3¿
5random_rotation/rotation_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        27
5random_rotation/rotation_matrix/strided_slice_4/stackÃ
7random_rotation/rotation_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation/rotation_matrix/strided_slice_4/stack_1Ã
7random_rotation/rotation_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7random_rotation/rotation_matrix/strided_slice_4/stack_2×
/random_rotation/rotation_matrix/strided_slice_4StridedSlice)random_rotation/rotation_matrix/Sin_3:y:0>random_rotation/rotation_matrix/strided_slice_4/stack:output:0@random_rotation/rotation_matrix/strided_slice_4/stack_1:output:0@random_rotation/rotation_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask21
/random_rotation/rotation_matrix/strided_slice_4©
%random_rotation/rotation_matrix/Cos_3Cos$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%random_rotation/rotation_matrix/Cos_3¿
5random_rotation/rotation_matrix/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        27
5random_rotation/rotation_matrix/strided_slice_5/stackÃ
7random_rotation/rotation_matrix/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation/rotation_matrix/strided_slice_5/stack_1Ã
7random_rotation/rotation_matrix/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7random_rotation/rotation_matrix/strided_slice_5/stack_2×
/random_rotation/rotation_matrix/strided_slice_5StridedSlice)random_rotation/rotation_matrix/Cos_3:y:0>random_rotation/rotation_matrix/strided_slice_5/stack:output:0@random_rotation/rotation_matrix/strided_slice_5/stack_1:output:0@random_rotation/rotation_matrix/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask21
/random_rotation/rotation_matrix/strided_slice_5¿
5random_rotation/rotation_matrix/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        27
5random_rotation/rotation_matrix/strided_slice_6/stackÃ
7random_rotation/rotation_matrix/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation/rotation_matrix/strided_slice_6/stack_1Ã
7random_rotation/rotation_matrix/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7random_rotation/rotation_matrix/strided_slice_6/stack_2Û
/random_rotation/rotation_matrix/strided_slice_6StridedSlice-random_rotation/rotation_matrix/truediv_1:z:0>random_rotation/rotation_matrix/strided_slice_6/stack:output:0@random_rotation/rotation_matrix/strided_slice_6/stack_1:output:0@random_rotation/rotation_matrix/strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask21
/random_rotation/rotation_matrix/strided_slice_6
+random_rotation/rotation_matrix/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2-
+random_rotation/rotation_matrix/zeros/mul/yì
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
B :è2.
,random_rotation/rotation_matrix/zeros/Less/yç
*random_rotation/rotation_matrix/zeros/LessLess-random_rotation/rotation_matrix/zeros/mul:z:05random_rotation/rotation_matrix/zeros/Less/y:output:0*
T0*
_output_shapes
: 2,
*random_rotation/rotation_matrix/zeros/Less¢
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
+random_rotation/rotation_matrix/zeros/Constõ
%random_rotation/rotation_matrix/zerosFill5random_rotation/rotation_matrix/zeros/packed:output:04random_rotation/rotation_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%random_rotation/rotation_matrix/zeros
+random_rotation/rotation_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2-
+random_rotation/rotation_matrix/concat/axisÈ
&random_rotation/rotation_matrix/concatConcatV28random_rotation/rotation_matrix/strided_slice_1:output:0'random_rotation/rotation_matrix/Neg:y:08random_rotation/rotation_matrix/strided_slice_3:output:08random_rotation/rotation_matrix/strided_slice_4:output:08random_rotation/rotation_matrix/strided_slice_5:output:08random_rotation/rotation_matrix/strided_slice_6:output:0.random_rotation/rotation_matrix/zeros:output:04random_rotation/rotation_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
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
/random_rotation/transform/strided_slice/stack_2ê
'random_rotation/transform/strided_sliceStridedSlice(random_rotation/transform/Shape:output:06random_rotation/transform/strided_slice/stack:output:08random_rotation/transform/strided_slice/stack_1:output:08random_rotation/transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2)
'random_rotation/transform/strided_slice
4random_rotation/transform/ImageProjectiveTransformV2ImageProjectiveTransformV2*random_flip/random_flip_left_right/add:z:0/random_rotation/rotation_matrix/concat:output:00random_rotation/transform/strided_slice:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
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
 *ÍÌ?2$
"random_contrast/random_uniform/maxÊ
,random_contrast/random_uniform/RandomUniformRandomUniform-random_contrast/random_uniform/shape:output:0*
T0*
_output_shapes
: *
dtype02.
,random_contrast/random_uniform/RandomUniformÊ
"random_contrast/random_uniform/subSub+random_contrast/random_uniform/max:output:0+random_contrast/random_uniform/min:output:0*
T0*
_output_shapes
: 2$
"random_contrast/random_uniform/subÏ
"random_contrast/random_uniform/mulMul5random_contrast/random_uniform/RandomUniform:output:0&random_contrast/random_uniform/sub:z:0*
T0*
_output_shapes
: 2$
"random_contrast/random_uniform/mul½
random_contrast/random_uniformAdd&random_contrast/random_uniform/mul:z:0+random_contrast/random_uniform/min:output:0*
T0*
_output_shapes
: 2 
random_contrast/random_uniformö
random_contrast/adjust_contrastAdjustContrastv2Irandom_rotation/transform/ImageProjectiveTransformV2:transformed_images:0"random_contrast/random_uniform:z:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
random_contrast/adjust_contrastÄ
(random_contrast/adjust_contrast/IdentityIdentity(random_contrast/adjust_contrast:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(random_contrast/adjust_contrast/IdentityÀ
IdentityIdentity1random_contrast/adjust_contrast/Identity:output:01^random_rotation/stateful_uniform/StatefulUniform*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ:2d
0random_rotation/stateful_uniform/StatefulUniform0random_rotation/stateful_uniform/StatefulUniform:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
`
æ
H__inference_random_flip_layer_call_and_return_conditional_losses_1737835

inputs
identity¢9random_flip_left_right/assert_greater_equal/Assert/Assert¢@random_flip_left_right/assert_positive/assert_less/Assert/Assertr
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
ýÿÿÿÿÿÿÿÿ2,
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
,random_flip_left_right/strided_slice/stack_2è
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
7random_flip_left_right/assert_positive/assert_less/Less¾
8random_flip_left_right/assert_positive/assert_less/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2:
8random_flip_left_right/assert_positive/assert_less/Const
6random_flip_left_right/assert_positive/assert_less/AllAll;random_flip_left_right/assert_positive/assert_less/Less:z:0Arandom_flip_left_right/assert_positive/assert_less/Const:output:0*
_output_shapes
: 28
6random_flip_left_right/assert_positive/assert_less/Allí
?random_flip_left_right/assert_positive/assert_less/Assert/ConstConst*
_output_shapes
: *
dtype0*;
value2B0 B*inner 3 dims of 'image.shape' must be > 0.2A
?random_flip_left_right/assert_positive/assert_less/Assert/Constý
Grandom_flip_left_right/assert_positive/assert_less/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*;
value2B0 B*inner 3 dims of 'image.shape' must be > 0.2I
Grandom_flip_left_right/assert_positive/assert_less/Assert/Assert/data_0Ã
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
1random_flip_left_right/assert_greater_equal/Constû
/random_flip_left_right/assert_greater_equal/AllAll<random_flip_left_right/assert_greater_equal/GreaterEqual:z:0:random_flip_left_right/assert_greater_equal/Const:output:0*
_output_shapes
: 21
/random_flip_left_right/assert_greater_equal/Allà
8random_flip_left_right/assert_greater_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*<
value3B1 B+'image' must be at least three-dimensional.2:
8random_flip_left_right/assert_greater_equal/Assert/Constä
:random_flip_left_right/assert_greater_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:2<
:random_flip_left_right/assert_greater_equal/Assert/Const_1Ý
:random_flip_left_right/assert_greater_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*5
value,B* B$x (random_flip_left_right/Rank:0) = 2<
:random_flip_left_right/assert_greater_equal/Assert/Const_2ï
:random_flip_left_right/assert_greater_equal/Assert/Const_3Const*
_output_shapes
: *
dtype0*G
value>B< B6y (random_flip_left_right/assert_greater_equal/y:0) = 2<
:random_flip_left_right/assert_greater_equal/Assert/Const_3ð
@random_flip_left_right/assert_greater_equal/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*<
value3B1 B+'image' must be at least three-dimensional.2B
@random_flip_left_right/assert_greater_equal/Assert/Assert/data_0ð
@random_flip_left_right/assert_greater_equal/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:2B
@random_flip_left_right/assert_greater_equal/Assert/Assert/data_1é
@random_flip_left_right/assert_greater_equal/Assert/Assert/data_2Const*
_output_shapes
: *
dtype0*5
value,B* B$x (random_flip_left_right/Rank:0) = 2B
@random_flip_left_right/assert_greater_equal/Assert/Assert/data_2û
@random_flip_left_right/assert_greater_equal/Assert/Assert/data_4Const*
_output_shapes
: *
dtype0*G
value>B< B6y (random_flip_left_right/assert_greater_equal/y:0) = 2B
@random_flip_left_right/assert_greater_equal/Assert/Assert/data_4®
9random_flip_left_right/assert_greater_equal/Assert/AssertAssert8random_flip_left_right/assert_greater_equal/All:output:0Irandom_flip_left_right/assert_greater_equal/Assert/Assert/data_0:output:0Irandom_flip_left_right/assert_greater_equal/Assert/Assert/data_1:output:0Irandom_flip_left_right/assert_greater_equal/Assert/Assert/data_2:output:0$random_flip_left_right/Rank:output:0Irandom_flip_left_right/assert_greater_equal/Assert/Assert/data_4:output:06random_flip_left_right/assert_greater_equal/y:output:0A^random_flip_left_right/assert_positive/assert_less/Assert/Assert*
T

2*
_output_shapes
 2;
9random_flip_left_right/assert_greater_equal/Assert/AssertÙ
)random_flip_left_right/control_dependencyIdentityinputs:^random_flip_left_right/assert_greater_equal/Assert/AssertA^random_flip_left_right/assert_positive/assert_less/Assert/Assert*
T0*
_class
loc:@inputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2+
)random_flip_left_right/control_dependency¢
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
,random_flip_left_right/strided_slice_1/stackª
.random_flip_left_right/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.random_flip_left_right/strided_slice_1/stack_1ª
.random_flip_left_right/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.random_flip_left_right/strided_slice_1/stack_2ø
&random_flip_left_right/strided_slice_1StridedSlice'random_flip_left_right/Shape_1:output:05random_flip_left_right/strided_slice_1/stack:output:07random_flip_left_right/strided_slice_1/stack_1:output:07random_flip_left_right/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&random_flip_left_right/strided_slice_1Á
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
)random_flip_left_right/random_uniform/maxì
3random_flip_left_right/random_uniform/RandomUniformRandomUniform4random_flip_left_right/random_uniform/shape:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype025
3random_flip_left_right/random_uniform/RandomUniformý
)random_flip_left_right/random_uniform/MulMul<random_flip_left_right/random_uniform/RandomUniform:output:02random_flip_left_right/random_uniform/max:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
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
&random_flip_left_right/Reshape/shape/3Æ
$random_flip_left_right/Reshape/shapePack/random_flip_left_right/strided_slice_1:output:0/random_flip_left_right/Reshape/shape/1:output:0/random_flip_left_right/Reshape/shape/2:output:0/random_flip_left_right/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2&
$random_flip_left_right/Reshape/shapeã
random_flip_left_right/ReshapeReshape-random_flip_left_right/random_uniform/Mul:z:0-random_flip_left_right/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
random_flip_left_right/Reshape¨
random_flip_left_right/RoundRound'random_flip_left_right/Reshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2"
 random_flip_left_right/ReverseV2á
random_flip_left_right/mulMul random_flip_left_right/Round:y:0)random_flip_left_right/ReverseV2:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
random_flip_left_right/mul
random_flip_left_right/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
random_flip_left_right/sub/xÂ
random_flip_left_right/subSub%random_flip_left_right/sub/x:output:0 random_flip_left_right/Round:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
random_flip_left_right/subì
random_flip_left_right/mul_1Mulrandom_flip_left_right/sub:z:02random_flip_left_right/control_dependency:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
random_flip_left_right/mul_1Ø
random_flip_left_right/addAddV2random_flip_left_right/mul:z:0 random_flip_left_right/mul_1:z:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
random_flip_left_right/add
IdentityIdentityrandom_flip_left_right/add:z:0:^random_flip_left_right/assert_greater_equal/Assert/AssertA^random_flip_left_right/assert_positive/assert_less/Assert/Assert*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2v
9random_flip_left_right/assert_greater_equal/Assert/Assert9random_flip_left_right/assert_greater_equal/Assert/Assert2
@random_flip_left_right/assert_positive/assert_less/Assert/Assert@random_flip_left_right/assert_positive/assert_less/Assert/Assert:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
§
I
-__inference_random_flip_layer_call_fn_1737849

inputs
identityé
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_random_flip_layer_call_and_return_conditional_losses_17366192
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
é
w
1__inference_random_rotation_layer_call_fn_1737966

inputs
unknown
identity¢StatefulPartitionedCallô
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_random_rotation_layer_call_and_return_conditional_losses_17367752
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ:22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ùñ

V__inference_SimpleModel_LessDense_avg_layer_call_and_return_conditional_losses_1737388

inputsX
Ttraining_data_augmentation_random_rotation_stateful_uniform_statefuluniform_resource-
)conv2d_101_conv2d_readvariableop_resource.
*conv2d_101_biasadd_readvariableop_resource,
(dense_201_matmul_readvariableop_resource-
)dense_201_biasadd_readvariableop_resource,
(dense_202_matmul_readvariableop_resource-
)dense_202_biasadd_readvariableop_resource,
(dense_203_matmul_readvariableop_resource-
)dense_203_biasadd_readvariableop_resource
identity¢KTraining_Data_Augmentation/random_rotation/stateful_uniform/StatefulUniform
PTraining_Data_Augmentation/random_flip/random_flip_left_right/control_dependencyIdentityinputs*
T0*
_class
loc:@inputs*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2R
PTraining_Data_Augmentation/random_flip/random_flip_left_right/control_dependency
CTraining_Data_Augmentation/random_flip/random_flip_left_right/ShapeShapeYTraining_Data_Augmentation/random_flip/random_flip_left_right/control_dependency:output:0*
T0*
_output_shapes
:2E
CTraining_Data_Augmentation/random_flip/random_flip_left_right/Shapeð
QTraining_Data_Augmentation/random_flip/random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2S
QTraining_Data_Augmentation/random_flip/random_flip_left_right/strided_slice/stackô
STraining_Data_Augmentation/random_flip/random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2U
STraining_Data_Augmentation/random_flip/random_flip_left_right/strided_slice/stack_1ô
STraining_Data_Augmentation/random_flip/random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2U
STraining_Data_Augmentation/random_flip/random_flip_left_right/strided_slice/stack_2Ö
KTraining_Data_Augmentation/random_flip/random_flip_left_right/strided_sliceStridedSliceLTraining_Data_Augmentation/random_flip/random_flip_left_right/Shape:output:0ZTraining_Data_Augmentation/random_flip/random_flip_left_right/strided_slice/stack:output:0\Training_Data_Augmentation/random_flip/random_flip_left_right/strided_slice/stack_1:output:0\Training_Data_Augmentation/random_flip/random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2M
KTraining_Data_Augmentation/random_flip/random_flip_left_right/strided_slice´
RTraining_Data_Augmentation/random_flip/random_flip_left_right/random_uniform/shapePackTTraining_Data_Augmentation/random_flip/random_flip_left_right/strided_slice:output:0*
N*
T0*
_output_shapes
:2T
RTraining_Data_Augmentation/random_flip/random_flip_left_right/random_uniform/shapeé
PTraining_Data_Augmentation/random_flip/random_flip_left_right/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2R
PTraining_Data_Augmentation/random_flip/random_flip_left_right/random_uniform/miné
PTraining_Data_Augmentation/random_flip/random_flip_left_right/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2R
PTraining_Data_Augmentation/random_flip/random_flip_left_right/random_uniform/maxá
ZTraining_Data_Augmentation/random_flip/random_flip_left_right/random_uniform/RandomUniformRandomUniform[Training_Data_Augmentation/random_flip/random_flip_left_right/random_uniform/shape:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02\
ZTraining_Data_Augmentation/random_flip/random_flip_left_right/random_uniform/RandomUniform
PTraining_Data_Augmentation/random_flip/random_flip_left_right/random_uniform/MulMulcTraining_Data_Augmentation/random_flip/random_flip_left_right/random_uniform/RandomUniform:output:0YTraining_Data_Augmentation/random_flip/random_flip_left_right/random_uniform/max:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2R
PTraining_Data_Augmentation/random_flip/random_flip_left_right/random_uniform/Mulà
MTraining_Data_Augmentation/random_flip/random_flip_left_right/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2O
MTraining_Data_Augmentation/random_flip/random_flip_left_right/Reshape/shape/1à
MTraining_Data_Augmentation/random_flip/random_flip_left_right/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2O
MTraining_Data_Augmentation/random_flip/random_flip_left_right/Reshape/shape/2à
MTraining_Data_Augmentation/random_flip/random_flip_left_right/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2O
MTraining_Data_Augmentation/random_flip/random_flip_left_right/Reshape/shape/3®
KTraining_Data_Augmentation/random_flip/random_flip_left_right/Reshape/shapePackTTraining_Data_Augmentation/random_flip/random_flip_left_right/strided_slice:output:0VTraining_Data_Augmentation/random_flip/random_flip_left_right/Reshape/shape/1:output:0VTraining_Data_Augmentation/random_flip/random_flip_left_right/Reshape/shape/2:output:0VTraining_Data_Augmentation/random_flip/random_flip_left_right/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2M
KTraining_Data_Augmentation/random_flip/random_flip_left_right/Reshape/shapeÿ
ETraining_Data_Augmentation/random_flip/random_flip_left_right/ReshapeReshapeTTraining_Data_Augmentation/random_flip/random_flip_left_right/random_uniform/Mul:z:0TTraining_Data_Augmentation/random_flip/random_flip_left_right/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2G
ETraining_Data_Augmentation/random_flip/random_flip_left_right/Reshape
CTraining_Data_Augmentation/random_flip/random_flip_left_right/RoundRoundNTraining_Data_Augmentation/random_flip/random_flip_left_right/Reshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2E
CTraining_Data_Augmentation/random_flip/random_flip_left_right/Roundæ
LTraining_Data_Augmentation/random_flip/random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:2N
LTraining_Data_Augmentation/random_flip/random_flip_left_right/ReverseV2/axis
GTraining_Data_Augmentation/random_flip/random_flip_left_right/ReverseV2	ReverseV2YTraining_Data_Augmentation/random_flip/random_flip_left_right/control_dependency:output:0UTraining_Data_Augmentation/random_flip/random_flip_left_right/ReverseV2/axis:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2I
GTraining_Data_Augmentation/random_flip/random_flip_left_right/ReverseV2â
ATraining_Data_Augmentation/random_flip/random_flip_left_right/mulMulGTraining_Data_Augmentation/random_flip/random_flip_left_right/Round:y:0PTraining_Data_Augmentation/random_flip/random_flip_left_right/ReverseV2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2C
ATraining_Data_Augmentation/random_flip/random_flip_left_right/mulÏ
CTraining_Data_Augmentation/random_flip/random_flip_left_right/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2E
CTraining_Data_Augmentation/random_flip/random_flip_left_right/sub/xÞ
ATraining_Data_Augmentation/random_flip/random_flip_left_right/subSubLTraining_Data_Augmentation/random_flip/random_flip_left_right/sub/x:output:0GTraining_Data_Augmentation/random_flip/random_flip_left_right/Round:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2C
ATraining_Data_Augmentation/random_flip/random_flip_left_right/subí
CTraining_Data_Augmentation/random_flip/random_flip_left_right/mul_1MulETraining_Data_Augmentation/random_flip/random_flip_left_right/sub:z:0YTraining_Data_Augmentation/random_flip/random_flip_left_right/control_dependency:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2E
CTraining_Data_Augmentation/random_flip/random_flip_left_right/mul_1Ù
ATraining_Data_Augmentation/random_flip/random_flip_left_right/addAddV2ETraining_Data_Augmentation/random_flip/random_flip_left_right/mul:z:0GTraining_Data_Augmentation/random_flip/random_flip_left_right/mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2C
ATraining_Data_Augmentation/random_flip/random_flip_left_right/addÙ
0Training_Data_Augmentation/random_rotation/ShapeShapeETraining_Data_Augmentation/random_flip/random_flip_left_right/add:z:0*
T0*
_output_shapes
:22
0Training_Data_Augmentation/random_rotation/ShapeÊ
>Training_Data_Augmentation/random_rotation/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2@
>Training_Data_Augmentation/random_rotation/strided_slice/stackÎ
@Training_Data_Augmentation/random_rotation/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2B
@Training_Data_Augmentation/random_rotation/strided_slice/stack_1Î
@Training_Data_Augmentation/random_rotation/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2B
@Training_Data_Augmentation/random_rotation/strided_slice/stack_2ä
8Training_Data_Augmentation/random_rotation/strided_sliceStridedSlice9Training_Data_Augmentation/random_rotation/Shape:output:0GTraining_Data_Augmentation/random_rotation/strided_slice/stack:output:0ITraining_Data_Augmentation/random_rotation/strided_slice/stack_1:output:0ITraining_Data_Augmentation/random_rotation/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2:
8Training_Data_Augmentation/random_rotation/strided_sliceÎ
@Training_Data_Augmentation/random_rotation/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2B
@Training_Data_Augmentation/random_rotation/strided_slice_1/stackÒ
BTraining_Data_Augmentation/random_rotation/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2D
BTraining_Data_Augmentation/random_rotation/strided_slice_1/stack_1Ò
BTraining_Data_Augmentation/random_rotation/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2D
BTraining_Data_Augmentation/random_rotation/strided_slice_1/stack_2î
:Training_Data_Augmentation/random_rotation/strided_slice_1StridedSlice9Training_Data_Augmentation/random_rotation/Shape:output:0ITraining_Data_Augmentation/random_rotation/strided_slice_1/stack:output:0KTraining_Data_Augmentation/random_rotation/strided_slice_1/stack_1:output:0KTraining_Data_Augmentation/random_rotation/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2<
:Training_Data_Augmentation/random_rotation/strided_slice_1ß
/Training_Data_Augmentation/random_rotation/CastCastCTraining_Data_Augmentation/random_rotation/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 21
/Training_Data_Augmentation/random_rotation/CastÎ
@Training_Data_Augmentation/random_rotation/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2B
@Training_Data_Augmentation/random_rotation/strided_slice_2/stackÒ
BTraining_Data_Augmentation/random_rotation/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2D
BTraining_Data_Augmentation/random_rotation/strided_slice_2/stack_1Ò
BTraining_Data_Augmentation/random_rotation/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2D
BTraining_Data_Augmentation/random_rotation/strided_slice_2/stack_2î
:Training_Data_Augmentation/random_rotation/strided_slice_2StridedSlice9Training_Data_Augmentation/random_rotation/Shape:output:0ITraining_Data_Augmentation/random_rotation/strided_slice_2/stack:output:0KTraining_Data_Augmentation/random_rotation/strided_slice_2/stack_1:output:0KTraining_Data_Augmentation/random_rotation/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2<
:Training_Data_Augmentation/random_rotation/strided_slice_2ã
1Training_Data_Augmentation/random_rotation/Cast_1CastCTraining_Data_Augmentation/random_rotation/strided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: 23
1Training_Data_Augmentation/random_rotation/Cast_1ÿ
ATraining_Data_Augmentation/random_rotation/stateful_uniform/shapePackATraining_Data_Augmentation/random_rotation/strided_slice:output:0*
N*
T0*
_output_shapes
:2C
ATraining_Data_Augmentation/random_rotation/stateful_uniform/shapeÇ
?Training_Data_Augmentation/random_rotation/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *|Ù ¿2A
?Training_Data_Augmentation/random_rotation/stateful_uniform/minÇ
?Training_Data_Augmentation/random_rotation/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *|Ù ?2A
?Training_Data_Augmentation/random_rotation/stateful_uniform/maxð
UTraining_Data_Augmentation/random_rotation/stateful_uniform/StatefulUniform/algorithmConst*
_output_shapes
: *
dtype0	*
value	B	 R2W
UTraining_Data_Augmentation/random_rotation/stateful_uniform/StatefulUniform/algorithmç
KTraining_Data_Augmentation/random_rotation/stateful_uniform/StatefulUniformStatefulUniformTtraining_data_augmentation_random_rotation_stateful_uniform_statefuluniform_resource^Training_Data_Augmentation/random_rotation/stateful_uniform/StatefulUniform/algorithm:output:0JTraining_Data_Augmentation/random_rotation/stateful_uniform/shape:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shape_dtype02M
KTraining_Data_Augmentation/random_rotation/stateful_uniform/StatefulUniform¾
?Training_Data_Augmentation/random_rotation/stateful_uniform/subSubHTraining_Data_Augmentation/random_rotation/stateful_uniform/max:output:0HTraining_Data_Augmentation/random_rotation/stateful_uniform/min:output:0*
T0*
_output_shapes
: 2A
?Training_Data_Augmentation/random_rotation/stateful_uniform/subÒ
?Training_Data_Augmentation/random_rotation/stateful_uniform/mulMulTTraining_Data_Augmentation/random_rotation/stateful_uniform/StatefulUniform:output:0CTraining_Data_Augmentation/random_rotation/stateful_uniform/sub:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2A
?Training_Data_Augmentation/random_rotation/stateful_uniform/mul¾
;Training_Data_Augmentation/random_rotation/stateful_uniformAddCTraining_Data_Augmentation/random_rotation/stateful_uniform/mul:z:0HTraining_Data_Augmentation/random_rotation/stateful_uniform/min:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2=
;Training_Data_Augmentation/random_rotation/stateful_uniformÉ
@Training_Data_Augmentation/random_rotation/rotation_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2B
@Training_Data_Augmentation/random_rotation/rotation_matrix/sub/yª
>Training_Data_Augmentation/random_rotation/rotation_matrix/subSub5Training_Data_Augmentation/random_rotation/Cast_1:y:0ITraining_Data_Augmentation/random_rotation/rotation_matrix/sub/y:output:0*
T0*
_output_shapes
: 2@
>Training_Data_Augmentation/random_rotation/rotation_matrix/subö
>Training_Data_Augmentation/random_rotation/rotation_matrix/CosCos?Training_Data_Augmentation/random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2@
>Training_Data_Augmentation/random_rotation/rotation_matrix/CosÍ
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
@Training_Data_Augmentation/random_rotation/rotation_matrix/sub_1¿
>Training_Data_Augmentation/random_rotation/rotation_matrix/mulMulBTraining_Data_Augmentation/random_rotation/rotation_matrix/Cos:y:0DTraining_Data_Augmentation/random_rotation/rotation_matrix/sub_1:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2@
>Training_Data_Augmentation/random_rotation/rotation_matrix/mulö
>Training_Data_Augmentation/random_rotation/rotation_matrix/SinSin?Training_Data_Augmentation/random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2@
>Training_Data_Augmentation/random_rotation/rotation_matrix/SinÍ
BTraining_Data_Augmentation/random_rotation/rotation_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2D
BTraining_Data_Augmentation/random_rotation/rotation_matrix/sub_2/y®
@Training_Data_Augmentation/random_rotation/rotation_matrix/sub_2Sub3Training_Data_Augmentation/random_rotation/Cast:y:0KTraining_Data_Augmentation/random_rotation/rotation_matrix/sub_2/y:output:0*
T0*
_output_shapes
: 2B
@Training_Data_Augmentation/random_rotation/rotation_matrix/sub_2Ã
@Training_Data_Augmentation/random_rotation/rotation_matrix/mul_1MulBTraining_Data_Augmentation/random_rotation/rotation_matrix/Sin:y:0DTraining_Data_Augmentation/random_rotation/rotation_matrix/sub_2:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2B
@Training_Data_Augmentation/random_rotation/rotation_matrix/mul_1Ã
@Training_Data_Augmentation/random_rotation/rotation_matrix/sub_3SubBTraining_Data_Augmentation/random_rotation/rotation_matrix/mul:z:0DTraining_Data_Augmentation/random_rotation/rotation_matrix/mul_1:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2B
@Training_Data_Augmentation/random_rotation/rotation_matrix/sub_3Ã
@Training_Data_Augmentation/random_rotation/rotation_matrix/sub_4SubBTraining_Data_Augmentation/random_rotation/rotation_matrix/sub:z:0DTraining_Data_Augmentation/random_rotation/rotation_matrix/sub_3:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2B
@Training_Data_Augmentation/random_rotation/rotation_matrix/sub_4Ñ
DTraining_Data_Augmentation/random_rotation/rotation_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2F
DTraining_Data_Augmentation/random_rotation/rotation_matrix/truediv/yÖ
BTraining_Data_Augmentation/random_rotation/rotation_matrix/truedivRealDivDTraining_Data_Augmentation/random_rotation/rotation_matrix/sub_4:z:0MTraining_Data_Augmentation/random_rotation/rotation_matrix/truediv/y:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2D
BTraining_Data_Augmentation/random_rotation/rotation_matrix/truedivÍ
BTraining_Data_Augmentation/random_rotation/rotation_matrix/sub_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2D
BTraining_Data_Augmentation/random_rotation/rotation_matrix/sub_5/y®
@Training_Data_Augmentation/random_rotation/rotation_matrix/sub_5Sub3Training_Data_Augmentation/random_rotation/Cast:y:0KTraining_Data_Augmentation/random_rotation/rotation_matrix/sub_5/y:output:0*
T0*
_output_shapes
: 2B
@Training_Data_Augmentation/random_rotation/rotation_matrix/sub_5ú
@Training_Data_Augmentation/random_rotation/rotation_matrix/Sin_1Sin?Training_Data_Augmentation/random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2B
@Training_Data_Augmentation/random_rotation/rotation_matrix/Sin_1Í
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
@Training_Data_Augmentation/random_rotation/rotation_matrix/sub_6Å
@Training_Data_Augmentation/random_rotation/rotation_matrix/mul_2MulDTraining_Data_Augmentation/random_rotation/rotation_matrix/Sin_1:y:0DTraining_Data_Augmentation/random_rotation/rotation_matrix/sub_6:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2B
@Training_Data_Augmentation/random_rotation/rotation_matrix/mul_2ú
@Training_Data_Augmentation/random_rotation/rotation_matrix/Cos_1Cos?Training_Data_Augmentation/random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2B
@Training_Data_Augmentation/random_rotation/rotation_matrix/Cos_1Í
BTraining_Data_Augmentation/random_rotation/rotation_matrix/sub_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2D
BTraining_Data_Augmentation/random_rotation/rotation_matrix/sub_7/y®
@Training_Data_Augmentation/random_rotation/rotation_matrix/sub_7Sub3Training_Data_Augmentation/random_rotation/Cast:y:0KTraining_Data_Augmentation/random_rotation/rotation_matrix/sub_7/y:output:0*
T0*
_output_shapes
: 2B
@Training_Data_Augmentation/random_rotation/rotation_matrix/sub_7Å
@Training_Data_Augmentation/random_rotation/rotation_matrix/mul_3MulDTraining_Data_Augmentation/random_rotation/rotation_matrix/Cos_1:y:0DTraining_Data_Augmentation/random_rotation/rotation_matrix/sub_7:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2B
@Training_Data_Augmentation/random_rotation/rotation_matrix/mul_3Ã
>Training_Data_Augmentation/random_rotation/rotation_matrix/addAddV2DTraining_Data_Augmentation/random_rotation/rotation_matrix/mul_2:z:0DTraining_Data_Augmentation/random_rotation/rotation_matrix/mul_3:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2@
>Training_Data_Augmentation/random_rotation/rotation_matrix/addÃ
@Training_Data_Augmentation/random_rotation/rotation_matrix/sub_8SubDTraining_Data_Augmentation/random_rotation/rotation_matrix/sub_5:z:0BTraining_Data_Augmentation/random_rotation/rotation_matrix/add:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2B
@Training_Data_Augmentation/random_rotation/rotation_matrix/sub_8Õ
FTraining_Data_Augmentation/random_rotation/rotation_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2H
FTraining_Data_Augmentation/random_rotation/rotation_matrix/truediv_1/yÜ
DTraining_Data_Augmentation/random_rotation/rotation_matrix/truediv_1RealDivDTraining_Data_Augmentation/random_rotation/rotation_matrix/sub_8:z:0OTraining_Data_Augmentation/random_rotation/rotation_matrix/truediv_1/y:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2F
DTraining_Data_Augmentation/random_rotation/rotation_matrix/truediv_1ó
@Training_Data_Augmentation/random_rotation/rotation_matrix/ShapeShape?Training_Data_Augmentation/random_rotation/stateful_uniform:z:0*
T0*
_output_shapes
:2B
@Training_Data_Augmentation/random_rotation/rotation_matrix/Shapeê
NTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2P
NTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice/stackî
PTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2R
PTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice/stack_1î
PTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2R
PTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice/stack_2Ä
HTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_sliceStridedSliceITraining_Data_Augmentation/random_rotation/rotation_matrix/Shape:output:0WTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice/stack:output:0YTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice/stack_1:output:0YTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2J
HTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_sliceú
@Training_Data_Augmentation/random_rotation/rotation_matrix/Cos_2Cos?Training_Data_Augmentation/random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2B
@Training_Data_Augmentation/random_rotation/rotation_matrix/Cos_2õ
PTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2R
PTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_1/stackù
RTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2T
RTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_1/stack_1ù
RTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2T
RTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_1/stack_2ù
JTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_1StridedSliceDTraining_Data_Augmentation/random_rotation/rotation_matrix/Cos_2:y:0YTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_1/stack:output:0[Training_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_1/stack_1:output:0[Training_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask2L
JTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_1ú
@Training_Data_Augmentation/random_rotation/rotation_matrix/Sin_2Sin?Training_Data_Augmentation/random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2B
@Training_Data_Augmentation/random_rotation/rotation_matrix/Sin_2õ
PTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2R
PTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_2/stackù
RTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2T
RTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_2/stack_1ù
RTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2T
RTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_2/stack_2ù
JTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_2StridedSliceDTraining_Data_Augmentation/random_rotation/rotation_matrix/Sin_2:y:0YTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_2/stack:output:0[Training_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_2/stack_1:output:0[Training_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask2L
JTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_2
>Training_Data_Augmentation/random_rotation/rotation_matrix/NegNegSTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2@
>Training_Data_Augmentation/random_rotation/rotation_matrix/Negõ
PTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        2R
PTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_3/stackù
RTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2T
RTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_3/stack_1ù
RTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2T
RTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_3/stack_2û
JTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_3StridedSliceFTraining_Data_Augmentation/random_rotation/rotation_matrix/truediv:z:0YTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_3/stack:output:0[Training_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_3/stack_1:output:0[Training_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask2L
JTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_3ú
@Training_Data_Augmentation/random_rotation/rotation_matrix/Sin_3Sin?Training_Data_Augmentation/random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2B
@Training_Data_Augmentation/random_rotation/rotation_matrix/Sin_3õ
PTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        2R
PTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_4/stackù
RTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2T
RTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_4/stack_1ù
RTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2T
RTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_4/stack_2ù
JTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_4StridedSliceDTraining_Data_Augmentation/random_rotation/rotation_matrix/Sin_3:y:0YTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_4/stack:output:0[Training_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_4/stack_1:output:0[Training_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask2L
JTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_4ú
@Training_Data_Augmentation/random_rotation/rotation_matrix/Cos_3Cos?Training_Data_Augmentation/random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2B
@Training_Data_Augmentation/random_rotation/rotation_matrix/Cos_3õ
PTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        2R
PTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_5/stackù
RTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2T
RTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_5/stack_1ù
RTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2T
RTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_5/stack_2ù
JTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_5StridedSliceDTraining_Data_Augmentation/random_rotation/rotation_matrix/Cos_3:y:0YTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_5/stack:output:0[Training_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_5/stack_1:output:0[Training_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask2L
JTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_5õ
PTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2R
PTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_6/stackù
RTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2T
RTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_6/stack_1ù
RTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2T
RTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_6/stack_2ý
JTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_6StridedSliceHTraining_Data_Augmentation/random_rotation/rotation_matrix/truediv_1:z:0YTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_6/stack:output:0[Training_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_6/stack_1:output:0[Training_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask2L
JTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_6Ò
FTraining_Data_Augmentation/random_rotation/rotation_matrix/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2H
FTraining_Data_Augmentation/random_rotation/rotation_matrix/zeros/mul/yØ
DTraining_Data_Augmentation/random_rotation/rotation_matrix/zeros/mulMulQTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice:output:0OTraining_Data_Augmentation/random_rotation/rotation_matrix/zeros/mul/y:output:0*
T0*
_output_shapes
: 2F
DTraining_Data_Augmentation/random_rotation/rotation_matrix/zeros/mulÕ
GTraining_Data_Augmentation/random_rotation/rotation_matrix/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2I
GTraining_Data_Augmentation/random_rotation/rotation_matrix/zeros/Less/yÓ
ETraining_Data_Augmentation/random_rotation/rotation_matrix/zeros/LessLessHTraining_Data_Augmentation/random_rotation/rotation_matrix/zeros/mul:z:0PTraining_Data_Augmentation/random_rotation/rotation_matrix/zeros/Less/y:output:0*
T0*
_output_shapes
: 2G
ETraining_Data_Augmentation/random_rotation/rotation_matrix/zeros/LessØ
ITraining_Data_Augmentation/random_rotation/rotation_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2K
ITraining_Data_Augmentation/random_rotation/rotation_matrix/zeros/packed/1ï
GTraining_Data_Augmentation/random_rotation/rotation_matrix/zeros/packedPackQTraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice:output:0RTraining_Data_Augmentation/random_rotation/rotation_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2I
GTraining_Data_Augmentation/random_rotation/rotation_matrix/zeros/packedÕ
FTraining_Data_Augmentation/random_rotation/rotation_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2H
FTraining_Data_Augmentation/random_rotation/rotation_matrix/zeros/Constá
@Training_Data_Augmentation/random_rotation/rotation_matrix/zerosFillPTraining_Data_Augmentation/random_rotation/rotation_matrix/zeros/packed:output:0OTraining_Data_Augmentation/random_rotation/rotation_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2B
@Training_Data_Augmentation/random_rotation/rotation_matrix/zerosÒ
FTraining_Data_Augmentation/random_rotation/rotation_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2H
FTraining_Data_Augmentation/random_rotation/rotation_matrix/concat/axisÖ
ATraining_Data_Augmentation/random_rotation/rotation_matrix/concatConcatV2STraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_1:output:0BTraining_Data_Augmentation/random_rotation/rotation_matrix/Neg:y:0STraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_3:output:0STraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_4:output:0STraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_5:output:0STraining_Data_Augmentation/random_rotation/rotation_matrix/strided_slice_6:output:0ITraining_Data_Augmentation/random_rotation/rotation_matrix/zeros:output:0OTraining_Data_Augmentation/random_rotation/rotation_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2C
ATraining_Data_Augmentation/random_rotation/rotation_matrix/concatí
:Training_Data_Augmentation/random_rotation/transform/ShapeShapeETraining_Data_Augmentation/random_flip/random_flip_left_right/add:z:0*
T0*
_output_shapes
:2<
:Training_Data_Augmentation/random_rotation/transform/ShapeÞ
HTraining_Data_Augmentation/random_rotation/transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2J
HTraining_Data_Augmentation/random_rotation/transform/strided_slice/stackâ
JTraining_Data_Augmentation/random_rotation/transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2L
JTraining_Data_Augmentation/random_rotation/transform/strided_slice/stack_1â
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
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR2Q
OTraining_Data_Augmentation/random_rotation/transform/ImageProjectiveTransformV2Å
?Training_Data_Augmentation/random_contrast/random_uniform/shapeConst*
_output_shapes
: *
dtype0*
valueB 2A
?Training_Data_Augmentation/random_contrast/random_uniform/shapeÃ
=Training_Data_Augmentation/random_contrast/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *fff?2?
=Training_Data_Augmentation/random_contrast/random_uniform/minÃ
=Training_Data_Augmentation/random_contrast/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌ?2?
=Training_Data_Augmentation/random_contrast/random_uniform/max
GTraining_Data_Augmentation/random_contrast/random_uniform/RandomUniformRandomUniformHTraining_Data_Augmentation/random_contrast/random_uniform/shape:output:0*
T0*
_output_shapes
: *
dtype02I
GTraining_Data_Augmentation/random_contrast/random_uniform/RandomUniform¶
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
9Training_Data_Augmentation/random_contrast/random_uniformâ
:Training_Data_Augmentation/random_contrast/adjust_contrastAdjustContrastv2dTraining_Data_Augmentation/random_rotation/transform/ImageProjectiveTransformV2:transformed_images:0=Training_Data_Augmentation/random_contrast/random_uniform:z:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2<
:Training_Data_Augmentation/random_contrast/adjust_contrast
CTraining_Data_Augmentation/random_contrast/adjust_contrast/IdentityIdentityCTraining_Data_Augmentation/random_contrast/adjust_contrast:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2E
CTraining_Data_Augmentation/random_contrast/adjust_contrast/Identity¶
 conv2d_101/Conv2D/ReadVariableOpReadVariableOp)conv2d_101_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02"
 conv2d_101/Conv2D/ReadVariableOp
conv2d_101/Conv2DConv2DLTraining_Data_Augmentation/random_contrast/adjust_contrast/Identity:output:0(conv2d_101/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
conv2d_101/Conv2D­
!conv2d_101/BiasAdd/ReadVariableOpReadVariableOp*conv2d_101_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02#
!conv2d_101/BiasAdd/ReadVariableOp´
conv2d_101/BiasAddBiasAddconv2d_101/Conv2D:output:0)conv2d_101/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d_101/BiasAdd
conv2d_101/ReluReluconv2d_101/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d_101/ReluÜ
average_pooling2d_46/AvgPoolAvgPoolconv2d_101/Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
2
average_pooling2d_46/AvgPool{
dropout_101/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?2
dropout_101/dropout/Const¾
dropout_101/dropout/MulMul%average_pooling2d_46/AvgPool:output:0"dropout_101/dropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_101/dropout/Mul
dropout_101/dropout/ShapeShape%average_pooling2d_46/AvgPool:output:0*
T0*
_output_shapes
:2
dropout_101/dropout/Shapeà
0dropout_101/dropout/random_uniform/RandomUniformRandomUniform"dropout_101/dropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype022
0dropout_101/dropout/random_uniform/RandomUniform
"dropout_101/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2$
"dropout_101/dropout/GreaterEqual/yö
 dropout_101/dropout/GreaterEqualGreaterEqual9dropout_101/dropout/random_uniform/RandomUniform:output:0+dropout_101/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2"
 dropout_101/dropout/GreaterEqual«
dropout_101/dropout/CastCast$dropout_101/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_101/dropout/Cast²
dropout_101/dropout/Mul_1Muldropout_101/dropout/Mul:z:0dropout_101/dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_101/dropout/Mul_1u
flatten_67/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@*  2
flatten_67/Const 
flatten_67/ReshapeReshapedropout_101/dropout/Mul_1:z:0flatten_67/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀT2
flatten_67/Reshape­
dense_201/MatMul/ReadVariableOpReadVariableOp(dense_201_matmul_readvariableop_resource* 
_output_shapes
:
ÀT*
dtype02!
dense_201/MatMul/ReadVariableOp§
dense_201/MatMulMatMulflatten_67/Reshape:output:0'dense_201/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_201/MatMul«
 dense_201/BiasAdd/ReadVariableOpReadVariableOp)dense_201_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 dense_201/BiasAdd/ReadVariableOpª
dense_201/BiasAddBiasAdddense_201/MatMul:product:0(dense_201/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_201/BiasAddw
dense_201/ReluReludense_201/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_201/Relu¬
dense_202/MatMul/ReadVariableOpReadVariableOp(dense_202_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02!
dense_202/MatMul/ReadVariableOp§
dense_202/MatMulMatMuldense_201/Relu:activations:0'dense_202/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_202/MatMulª
 dense_202/BiasAdd/ReadVariableOpReadVariableOp)dense_202_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 dense_202/BiasAdd/ReadVariableOp©
dense_202/BiasAddBiasAdddense_202/MatMul:product:0(dense_202/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_202/BiasAddv
dense_202/ReluReludense_202/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_202/Relu«
dense_203/MatMul/ReadVariableOpReadVariableOp(dense_203_matmul_readvariableop_resource*
_output_shapes

:@
*
dtype02!
dense_203/MatMul/ReadVariableOp§
dense_203/MatMulMatMuldense_202/Relu:activations:0'dense_203/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_203/MatMulª
 dense_203/BiasAdd/ReadVariableOpReadVariableOp)dense_203_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02"
 dense_203/BiasAdd/ReadVariableOp©
dense_203/BiasAddBiasAdddense_203/MatMul:product:0(dense_203/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_203/BiasAdd
dense_203/SoftmaxSoftmaxdense_203/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_203/Softmax½
IdentityIdentitydense_203/Softmax:softmax:0L^Training_Data_Augmentation/random_rotation/stateful_uniform/StatefulUniform*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ:::::::::2
KTraining_Data_Augmentation/random_rotation/stateful_uniform/StatefulUniformKTraining_Data_Augmentation/random_rotation/stateful_uniform/StatefulUniform:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
³
®
F__inference_dense_203_layer_call_and_return_conditional_losses_1737031

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
:ÿÿÿÿÿÿÿÿÿ
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
:ÿÿÿÿÿÿÿÿÿ
2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
»
k
L__inference_random_contrast_layer_call_and_return_conditional_losses_1737984

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
 *ÍÌ?2
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
:ÿÿÿÿÿÿÿÿÿ2
adjust_contrast
adjust_contrast/IdentityIdentityadjust_contrast:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
adjust_contrast/Identity}
IdentityIdentity!adjust_contrast/Identity:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
`
æ
H__inference_random_flip_layer_call_and_return_conditional_losses_1736610

inputs
identity¢9random_flip_left_right/assert_greater_equal/Assert/Assert¢@random_flip_left_right/assert_positive/assert_less/Assert/Assertr
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
ýÿÿÿÿÿÿÿÿ2,
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
,random_flip_left_right/strided_slice/stack_2è
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
7random_flip_left_right/assert_positive/assert_less/Less¾
8random_flip_left_right/assert_positive/assert_less/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2:
8random_flip_left_right/assert_positive/assert_less/Const
6random_flip_left_right/assert_positive/assert_less/AllAll;random_flip_left_right/assert_positive/assert_less/Less:z:0Arandom_flip_left_right/assert_positive/assert_less/Const:output:0*
_output_shapes
: 28
6random_flip_left_right/assert_positive/assert_less/Allí
?random_flip_left_right/assert_positive/assert_less/Assert/ConstConst*
_output_shapes
: *
dtype0*;
value2B0 B*inner 3 dims of 'image.shape' must be > 0.2A
?random_flip_left_right/assert_positive/assert_less/Assert/Constý
Grandom_flip_left_right/assert_positive/assert_less/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*;
value2B0 B*inner 3 dims of 'image.shape' must be > 0.2I
Grandom_flip_left_right/assert_positive/assert_less/Assert/Assert/data_0Ã
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
1random_flip_left_right/assert_greater_equal/Constû
/random_flip_left_right/assert_greater_equal/AllAll<random_flip_left_right/assert_greater_equal/GreaterEqual:z:0:random_flip_left_right/assert_greater_equal/Const:output:0*
_output_shapes
: 21
/random_flip_left_right/assert_greater_equal/Allà
8random_flip_left_right/assert_greater_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*<
value3B1 B+'image' must be at least three-dimensional.2:
8random_flip_left_right/assert_greater_equal/Assert/Constä
:random_flip_left_right/assert_greater_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:2<
:random_flip_left_right/assert_greater_equal/Assert/Const_1Ý
:random_flip_left_right/assert_greater_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*5
value,B* B$x (random_flip_left_right/Rank:0) = 2<
:random_flip_left_right/assert_greater_equal/Assert/Const_2ï
:random_flip_left_right/assert_greater_equal/Assert/Const_3Const*
_output_shapes
: *
dtype0*G
value>B< B6y (random_flip_left_right/assert_greater_equal/y:0) = 2<
:random_flip_left_right/assert_greater_equal/Assert/Const_3ð
@random_flip_left_right/assert_greater_equal/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*<
value3B1 B+'image' must be at least three-dimensional.2B
@random_flip_left_right/assert_greater_equal/Assert/Assert/data_0ð
@random_flip_left_right/assert_greater_equal/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:2B
@random_flip_left_right/assert_greater_equal/Assert/Assert/data_1é
@random_flip_left_right/assert_greater_equal/Assert/Assert/data_2Const*
_output_shapes
: *
dtype0*5
value,B* B$x (random_flip_left_right/Rank:0) = 2B
@random_flip_left_right/assert_greater_equal/Assert/Assert/data_2û
@random_flip_left_right/assert_greater_equal/Assert/Assert/data_4Const*
_output_shapes
: *
dtype0*G
value>B< B6y (random_flip_left_right/assert_greater_equal/y:0) = 2B
@random_flip_left_right/assert_greater_equal/Assert/Assert/data_4®
9random_flip_left_right/assert_greater_equal/Assert/AssertAssert8random_flip_left_right/assert_greater_equal/All:output:0Irandom_flip_left_right/assert_greater_equal/Assert/Assert/data_0:output:0Irandom_flip_left_right/assert_greater_equal/Assert/Assert/data_1:output:0Irandom_flip_left_right/assert_greater_equal/Assert/Assert/data_2:output:0$random_flip_left_right/Rank:output:0Irandom_flip_left_right/assert_greater_equal/Assert/Assert/data_4:output:06random_flip_left_right/assert_greater_equal/y:output:0A^random_flip_left_right/assert_positive/assert_less/Assert/Assert*
T

2*
_output_shapes
 2;
9random_flip_left_right/assert_greater_equal/Assert/AssertÙ
)random_flip_left_right/control_dependencyIdentityinputs:^random_flip_left_right/assert_greater_equal/Assert/AssertA^random_flip_left_right/assert_positive/assert_less/Assert/Assert*
T0*
_class
loc:@inputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2+
)random_flip_left_right/control_dependency¢
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
,random_flip_left_right/strided_slice_1/stackª
.random_flip_left_right/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.random_flip_left_right/strided_slice_1/stack_1ª
.random_flip_left_right/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.random_flip_left_right/strided_slice_1/stack_2ø
&random_flip_left_right/strided_slice_1StridedSlice'random_flip_left_right/Shape_1:output:05random_flip_left_right/strided_slice_1/stack:output:07random_flip_left_right/strided_slice_1/stack_1:output:07random_flip_left_right/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&random_flip_left_right/strided_slice_1Á
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
)random_flip_left_right/random_uniform/maxì
3random_flip_left_right/random_uniform/RandomUniformRandomUniform4random_flip_left_right/random_uniform/shape:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype025
3random_flip_left_right/random_uniform/RandomUniformý
)random_flip_left_right/random_uniform/MulMul<random_flip_left_right/random_uniform/RandomUniform:output:02random_flip_left_right/random_uniform/max:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
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
&random_flip_left_right/Reshape/shape/3Æ
$random_flip_left_right/Reshape/shapePack/random_flip_left_right/strided_slice_1:output:0/random_flip_left_right/Reshape/shape/1:output:0/random_flip_left_right/Reshape/shape/2:output:0/random_flip_left_right/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2&
$random_flip_left_right/Reshape/shapeã
random_flip_left_right/ReshapeReshape-random_flip_left_right/random_uniform/Mul:z:0-random_flip_left_right/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
random_flip_left_right/Reshape¨
random_flip_left_right/RoundRound'random_flip_left_right/Reshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2"
 random_flip_left_right/ReverseV2á
random_flip_left_right/mulMul random_flip_left_right/Round:y:0)random_flip_left_right/ReverseV2:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
random_flip_left_right/mul
random_flip_left_right/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
random_flip_left_right/sub/xÂ
random_flip_left_right/subSub%random_flip_left_right/sub/x:output:0 random_flip_left_right/Round:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
random_flip_left_right/subì
random_flip_left_right/mul_1Mulrandom_flip_left_right/sub:z:02random_flip_left_right/control_dependency:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
random_flip_left_right/mul_1Ø
random_flip_left_right/addAddV2random_flip_left_right/mul:z:0 random_flip_left_right/mul_1:z:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
random_flip_left_right/add
IdentityIdentityrandom_flip_left_right/add:z:0:^random_flip_left_right/assert_greater_equal/Assert/AssertA^random_flip_left_right/assert_positive/assert_less/Assert/Assert*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2v
9random_flip_left_right/assert_greater_equal/Assert/Assert9random_flip_left_right/assert_greater_equal/Assert/Assert2
@random_flip_left_right/assert_positive/assert_less/Assert/Assert@random_flip_left_right/assert_positive/assert_less/Assert/Assert:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ö
h
L__inference_random_contrast_layer_call_and_return_conditional_losses_1736813

inputs
identityb
IdentityIdentityinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
®
®
F__inference_dense_202_layer_call_and_return_conditional_losses_1737004

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
:ÿÿÿÿÿÿÿÿÿ@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¹
R
6__inference_average_pooling2d_46_layer_call_fn_1736875

inputs
identityò
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_average_pooling2d_46_layer_call_and_return_conditional_losses_17368692
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

s
W__inference_Training_Data_Augmentation_layer_call_and_return_conditional_losses_1737611

inputs
identityb
IdentityIdentityinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


,__inference_conv2d_101_layer_call_fn_1737643

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallÿ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_conv2d_101_layer_call_and_return_conditional_losses_17369052
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
º
I
-__inference_dropout_101_layer_call_fn_1737670

inputs
identityÎ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_101_layer_call_and_return_conditional_losses_17369392
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Â
M
1__inference_random_rotation_layer_call_fn_1737971

inputs
identityÒ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_random_rotation_layer_call_and_return_conditional_losses_17367792
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ë
f
H__inference_dropout_101_layer_call_and_return_conditional_losses_1736939

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
»
k
L__inference_random_contrast_layer_call_and_return_conditional_losses_1736809

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
 *ÍÌ?2
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
:ÿÿÿÿÿÿÿÿÿ2
adjust_contrast
adjust_contrast/IdentityIdentityadjust_contrast:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
adjust_contrast/Identity}
IdentityIdentity!adjust_contrast/Identity:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¾	

;__inference_SimpleModel_LessDense_avg_layer_call_fn_1737130$
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
identity¢StatefulPartitionedCallú
StatefulPartitionedCallStatefulPartitionedCall training_data_augmentation_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8 *_
fZRX
V__inference_SimpleModel_LessDense_avg_layer_call_and_return_conditional_losses_17371092
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ:::::::::22
StatefulPartitionedCallStatefulPartitionedCall:q m
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:
_user_specified_name" Training_Data_Augmentation_input
²
f
-__inference_random_flip_layer_call_fn_1737844

inputs
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_random_flip_layer_call_and_return_conditional_losses_17366102
StatefulPartitionedCall±
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
§$
À
V__inference_SimpleModel_LessDense_avg_layer_call_and_return_conditional_losses_1737160

inputs
conv2d_101_1737136
conv2d_101_1737138
dense_201_1737144
dense_201_1737146
dense_202_1737149
dense_202_1737151
dense_203_1737154
dense_203_1737156
identity¢"conv2d_101/StatefulPartitionedCall¢!dense_201/StatefulPartitionedCall¢!dense_202/StatefulPartitionedCall¢!dense_203/StatefulPartitionedCall
*Training_Data_Augmentation/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *`
f[RY
W__inference_Training_Data_Augmentation_layer_call_and_return_conditional_losses_17368602,
*Training_Data_Augmentation/PartitionedCallÖ
"conv2d_101/StatefulPartitionedCallStatefulPartitionedCall3Training_Data_Augmentation/PartitionedCall:output:0conv2d_101_1737136conv2d_101_1737138*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_conv2d_101_layer_call_and_return_conditional_losses_17369052$
"conv2d_101/StatefulPartitionedCall¦
$average_pooling2d_46/PartitionedCallPartitionedCall+conv2d_101/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_average_pooling2d_46_layer_call_and_return_conditional_losses_17368692&
$average_pooling2d_46/PartitionedCall
dropout_101/PartitionedCallPartitionedCall-average_pooling2d_46/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_101_layer_call_and_return_conditional_losses_17369392
dropout_101/PartitionedCallú
flatten_67/PartitionedCallPartitionedCall$dropout_101/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀT* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_67_layer_call_and_return_conditional_losses_17369582
flatten_67/PartitionedCallº
!dense_201/StatefulPartitionedCallStatefulPartitionedCall#flatten_67/PartitionedCall:output:0dense_201_1737144dense_201_1737146*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_201_layer_call_and_return_conditional_losses_17369772#
!dense_201/StatefulPartitionedCallÀ
!dense_202/StatefulPartitionedCallStatefulPartitionedCall*dense_201/StatefulPartitionedCall:output:0dense_202_1737149dense_202_1737151*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_202_layer_call_and_return_conditional_losses_17370042#
!dense_202/StatefulPartitionedCallÀ
!dense_203/StatefulPartitionedCallStatefulPartitionedCall*dense_202/StatefulPartitionedCall:output:0dense_203_1737154dense_203_1737156*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_203_layer_call_and_return_conditional_losses_17370312#
!dense_203/StatefulPartitionedCall
IdentityIdentity*dense_203/StatefulPartitionedCall:output:0#^conv2d_101/StatefulPartitionedCall"^dense_201/StatefulPartitionedCall"^dense_202/StatefulPartitionedCall"^dense_203/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ::::::::2H
"conv2d_101/StatefulPartitionedCall"conv2d_101/StatefulPartitionedCall2F
!dense_201/StatefulPartitionedCall!dense_201/StatefulPartitionedCall2F
!dense_202/StatefulPartitionedCall!dense_202/StatefulPartitionedCall2F
!dense_203/StatefulPartitionedCall!dense_203/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ð
ù
;__inference_SimpleModel_LessDense_avg_layer_call_fn_1737447

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
identity¢StatefulPartitionedCallà
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8 *_
fZRX
V__inference_SimpleModel_LessDense_avg_layer_call_and_return_conditional_losses_17371092
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ:::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ù
c
<__inference_Training_Data_Augmentation_layer_call_fn_1736863
random_flip_input
identityè
PartitionedCallPartitionedCallrandom_flip_input*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *`
f[RY
W__inference_Training_Data_Augmentation_layer_call_and_return_conditional_losses_17368602
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:b ^
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+
_user_specified_namerandom_flip_input
·
s
W__inference_Training_Data_Augmentation_layer_call_and_return_conditional_losses_1736860

inputs
identityæ
random_flip/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_random_flip_layer_call_and_return_conditional_losses_17366572
random_flip/PartitionedCall
random_rotation/PartitionedCallPartitionedCall$random_flip/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_random_rotation_layer_call_and_return_conditional_losses_17367792!
random_rotation/PartitionedCall
random_contrast/PartitionedCallPartitionedCall(random_rotation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_random_contrast_layer_call_and_return_conditional_losses_17368132!
random_contrast/PartitionedCall
IdentityIdentity(random_contrast/PartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¿
c
G__inference_flatten_67_layer_call_and_return_conditional_losses_1737676

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@*  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀT2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀT2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Î
j
1__inference_random_contrast_layer_call_fn_1737993

inputs
identity¢StatefulPartitionedCallê
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_random_contrast_layer_call_and_return_conditional_losses_17368092
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¤(
Ã
V__inference_SimpleModel_LessDense_avg_layer_call_and_return_conditional_losses_1737109

inputs&
"training_data_augmentation_1737082
conv2d_101_1737085
conv2d_101_1737087
dense_201_1737093
dense_201_1737095
dense_202_1737098
dense_202_1737100
dense_203_1737103
dense_203_1737105
identity¢2Training_Data_Augmentation/StatefulPartitionedCall¢"conv2d_101/StatefulPartitionedCall¢!dense_201/StatefulPartitionedCall¢!dense_202/StatefulPartitionedCall¢!dense_203/StatefulPartitionedCall¢#dropout_101/StatefulPartitionedCallÐ
2Training_Data_Augmentation/StatefulPartitionedCallStatefulPartitionedCallinputs"training_data_augmentation_1737082*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *`
f[RY
W__inference_Training_Data_Augmentation_layer_call_and_return_conditional_losses_173684624
2Training_Data_Augmentation/StatefulPartitionedCallÞ
"conv2d_101/StatefulPartitionedCallStatefulPartitionedCall;Training_Data_Augmentation/StatefulPartitionedCall:output:0conv2d_101_1737085conv2d_101_1737087*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_conv2d_101_layer_call_and_return_conditional_losses_17369052$
"conv2d_101/StatefulPartitionedCall¦
$average_pooling2d_46/PartitionedCallPartitionedCall+conv2d_101/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_average_pooling2d_46_layer_call_and_return_conditional_losses_17368692&
$average_pooling2d_46/PartitionedCall¥
#dropout_101/StatefulPartitionedCallStatefulPartitionedCall-average_pooling2d_46/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_101_layer_call_and_return_conditional_losses_17369342%
#dropout_101/StatefulPartitionedCall
flatten_67/PartitionedCallPartitionedCall,dropout_101/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀT* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_67_layer_call_and_return_conditional_losses_17369582
flatten_67/PartitionedCallº
!dense_201/StatefulPartitionedCallStatefulPartitionedCall#flatten_67/PartitionedCall:output:0dense_201_1737093dense_201_1737095*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_201_layer_call_and_return_conditional_losses_17369772#
!dense_201/StatefulPartitionedCallÀ
!dense_202/StatefulPartitionedCallStatefulPartitionedCall*dense_201/StatefulPartitionedCall:output:0dense_202_1737098dense_202_1737100*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_202_layer_call_and_return_conditional_losses_17370042#
!dense_202/StatefulPartitionedCallÀ
!dense_203/StatefulPartitionedCallStatefulPartitionedCall*dense_202/StatefulPartitionedCall:output:0dense_203_1737103dense_203_1737105*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_203_layer_call_and_return_conditional_losses_17370312#
!dense_203/StatefulPartitionedCallê
IdentityIdentity*dense_203/StatefulPartitionedCall:output:03^Training_Data_Augmentation/StatefulPartitionedCall#^conv2d_101/StatefulPartitionedCall"^dense_201/StatefulPartitionedCall"^dense_202/StatefulPartitionedCall"^dense_203/StatefulPartitionedCall$^dropout_101/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ:::::::::2h
2Training_Data_Augmentation/StatefulPartitionedCall2Training_Data_Augmentation/StatefulPartitionedCall2H
"conv2d_101/StatefulPartitionedCall"conv2d_101/StatefulPartitionedCall2F
!dense_201/StatefulPartitionedCall!dense_201/StatefulPartitionedCall2F
!dense_202/StatefulPartitionedCall!dense_202/StatefulPartitionedCall2F
!dense_203/StatefulPartitionedCall!dense_203/StatefulPartitionedCall2J
#dropout_101/StatefulPartitionedCall#dropout_101/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
³
®
F__inference_dense_203_layer_call_and_return_conditional_losses_1737732

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
:ÿÿÿÿÿÿÿÿÿ
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
:ÿÿÿÿÿÿÿÿÿ
2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
	
¯
G__inference_conv2d_101_layer_call_and_return_conditional_losses_1736905

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
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
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ:::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


W__inference_Training_Data_Augmentation_layer_call_and_return_conditional_losses_1736827
random_flip_input
random_rotation_1736793
identity¢'random_contrast/StatefulPartitionedCall¢#random_flip/StatefulPartitionedCall¢'random_rotation/StatefulPartitionedCall
#random_flip/StatefulPartitionedCallStatefulPartitionedCallrandom_flip_input*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_random_flip_layer_call_and_return_conditional_losses_17366532%
#random_flip/StatefulPartitionedCallÊ
'random_rotation/StatefulPartitionedCallStatefulPartitionedCall,random_flip/StatefulPartitionedCall:output:0random_rotation_1736793*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_random_rotation_layer_call_and_return_conditional_losses_17367752)
'random_rotation/StatefulPartitionedCallÚ
'random_contrast/StatefulPartitionedCallStatefulPartitionedCall0random_rotation/StatefulPartitionedCall:output:0$^random_flip/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_random_contrast_layer_call_and_return_conditional_losses_17368092)
'random_contrast/StatefulPartitionedCall
IdentityIdentity0random_contrast/StatefulPartitionedCall:output:0(^random_contrast/StatefulPartitionedCall$^random_flip/StatefulPartitionedCall(^random_rotation/StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ:2R
'random_contrast/StatefulPartitionedCall'random_contrast/StatefulPartitionedCall2J
#random_flip/StatefulPartitionedCall#random_flip/StatefulPartitionedCall2R
'random_rotation/StatefulPartitionedCall'random_rotation/StatefulPartitionedCall:b ^
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+
_user_specified_namerandom_flip_input
(
g
H__inference_random_flip_layer_call_and_return_conditional_losses_1736653

inputs
identity¿
)random_flip_left_right/control_dependencyIdentityinputs*
T0*
_class
loc:@inputs*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)random_flip_left_right/control_dependency
random_flip_left_right/ShapeShape2random_flip_left_right/control_dependency:output:0*
T0*
_output_shapes
:2
random_flip_left_right/Shape¢
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
,random_flip_left_right/strided_slice/stack_2ì
$random_flip_left_right/strided_sliceStridedSlice%random_flip_left_right/Shape:output:03random_flip_left_right/strided_slice/stack:output:05random_flip_left_right/strided_slice/stack_1:output:05random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$random_flip_left_right/strided_slice¿
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
)random_flip_left_right/random_uniform/maxì
3random_flip_left_right/random_uniform/RandomUniformRandomUniform4random_flip_left_right/random_uniform/shape:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype025
3random_flip_left_right/random_uniform/RandomUniformý
)random_flip_left_right/random_uniform/MulMul<random_flip_left_right/random_uniform/RandomUniform:output:02random_flip_left_right/random_uniform/max:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
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
&random_flip_left_right/Reshape/shape/3Ä
$random_flip_left_right/Reshape/shapePack-random_flip_left_right/strided_slice:output:0/random_flip_left_right/Reshape/shape/1:output:0/random_flip_left_right/Reshape/shape/2:output:0/random_flip_left_right/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2&
$random_flip_left_right/Reshape/shapeã
random_flip_left_right/ReshapeReshape-random_flip_left_right/random_uniform/Mul:z:0-random_flip_left_right/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
random_flip_left_right/Reshape¨
random_flip_left_right/RoundRound'random_flip_left_right/Reshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
random_flip_left_right/Round
%random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:2'
%random_flip_left_right/ReverseV2/axisï
 random_flip_left_right/ReverseV2	ReverseV22random_flip_left_right/control_dependency:output:0.random_flip_left_right/ReverseV2/axis:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 random_flip_left_right/ReverseV2Æ
random_flip_left_right/mulMul random_flip_left_right/Round:y:0)random_flip_left_right/ReverseV2:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
random_flip_left_right/mul
random_flip_left_right/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
random_flip_left_right/sub/xÂ
random_flip_left_right/subSub%random_flip_left_right/sub/x:output:0 random_flip_left_right/Round:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
random_flip_left_right/subÑ
random_flip_left_right/mul_1Mulrandom_flip_left_right/sub:z:02random_flip_left_right/control_dependency:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
random_flip_left_right/mul_1½
random_flip_left_right/addAddV2random_flip_left_right/mul:z:0 random_flip_left_right/mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
random_flip_left_right/addz
IdentityIdentityrandom_flip_left_right/add:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
²
½
L__inference_random_rotation_layer_call_and_return_conditional_losses_1737955

inputs-
)stateful_uniform_statefuluniform_resource
identity¢ stateful_uniform/StatefulUniformD
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
strided_slice/stack_2â
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
strided_slice_1/stack_2ì
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
strided_slice_2/stack_2ì
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
 *|Ù ¿2
stateful_uniform/minq
stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *|Ù ?2
stateful_uniform/max
*stateful_uniform/StatefulUniform/algorithmConst*
_output_shapes
: *
dtype0	*
value	B	 R2,
*stateful_uniform/StatefulUniform/algorithm
 stateful_uniform/StatefulUniformStatefulUniform)stateful_uniform_statefuluniform_resource3stateful_uniform/StatefulUniform/algorithm:output:0stateful_uniform/shape:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
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
:ÿÿÿÿÿÿÿÿÿ2
stateful_uniform/mul
stateful_uniformAddstateful_uniform/mul:z:0stateful_uniform/min:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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
:ÿÿÿÿÿÿÿÿÿ2
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
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/mulu
rotation_matrix/SinSinstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/mul_1
rotation_matrix/sub_3Subrotation_matrix/mul:z:0rotation_matrix/mul_1:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/sub_3
rotation_matrix/sub_4Subrotation_matrix/sub:z:0rotation_matrix/sub_3:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/sub_4{
rotation_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
rotation_matrix/truediv/yª
rotation_matrix/truedivRealDivrotation_matrix/sub_4:z:0"rotation_matrix/truediv/y:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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
:ÿÿÿÿÿÿÿÿÿ2
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
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/mul_2y
rotation_matrix/Cos_1Cosstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/mul_3
rotation_matrix/addAddV2rotation_matrix/mul_2:z:0rotation_matrix/mul_3:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/add
rotation_matrix/sub_8Subrotation_matrix/sub_5:z:0rotation_matrix/add:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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
:ÿÿÿÿÿÿÿÿÿ2
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
%rotation_matrix/strided_slice/stack_2Â
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
:ÿÿÿÿÿÿÿÿÿ2
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
'rotation_matrix/strided_slice_1/stack_2÷
rotation_matrix/strided_slice_1StridedSlicerotation_matrix/Cos_2:y:0.rotation_matrix/strided_slice_1/stack:output:00rotation_matrix/strided_slice_1/stack_1:output:00rotation_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_1y
rotation_matrix/Sin_2Sinstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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
'rotation_matrix/strided_slice_2/stack_2÷
rotation_matrix/strided_slice_2StridedSlicerotation_matrix/Sin_2:y:0.rotation_matrix/strided_slice_2/stack:output:00rotation_matrix/strided_slice_2/stack_1:output:00rotation_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_2
rotation_matrix/NegNeg(rotation_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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
'rotation_matrix/strided_slice_3/stack_2ù
rotation_matrix/strided_slice_3StridedSlicerotation_matrix/truediv:z:0.rotation_matrix/strided_slice_3/stack:output:00rotation_matrix/strided_slice_3/stack_1:output:00rotation_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_3y
rotation_matrix/Sin_3Sinstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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
'rotation_matrix/strided_slice_4/stack_2÷
rotation_matrix/strided_slice_4StridedSlicerotation_matrix/Sin_3:y:0.rotation_matrix/strided_slice_4/stack:output:00rotation_matrix/strided_slice_4/stack_1:output:00rotation_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_4y
rotation_matrix/Cos_3Cosstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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
'rotation_matrix/strided_slice_5/stack_2÷
rotation_matrix/strided_slice_5StridedSlicerotation_matrix/Cos_3:y:0.rotation_matrix/strided_slice_5/stack:output:00rotation_matrix/strided_slice_5/stack_1:output:00rotation_matrix/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

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
'rotation_matrix/strided_slice_6/stack_2û
rotation_matrix/strided_slice_6StridedSlicerotation_matrix/truediv_1:z:0.rotation_matrix/strided_slice_6/stack:output:00rotation_matrix/strided_slice_6/stack_1:output:00rotation_matrix/strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

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
B :è2
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
rotation_matrix/zeros/packed/1Ã
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
rotation_matrix/zeros/Constµ
rotation_matrix/zerosFill%rotation_matrix/zeros/packed:output:0$rotation_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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
:ÿÿÿÿÿÿÿÿÿ2
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
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR2&
$transform/ImageProjectiveTransformV2¸
IdentityIdentity9transform/ImageProjectiveTransformV2:transformed_images:0!^stateful_uniform/StatefulUniform*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ:2D
 stateful_uniform/StatefulUniform stateful_uniform/StatefulUniform:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
û

#__inference__traced_restore_1738251
file_prefix&
"assignvariableop_conv2d_101_kernel&
"assignvariableop_1_conv2d_101_bias'
#assignvariableop_2_dense_201_kernel%
!assignvariableop_3_dense_201_bias'
#assignvariableop_4_dense_202_kernel%
!assignvariableop_5_dense_202_bias'
#assignvariableop_6_dense_203_kernel%
!assignvariableop_7_dense_203_bias 
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
,assignvariableop_19_adam_conv2d_101_kernel_m.
*assignvariableop_20_adam_conv2d_101_bias_m/
+assignvariableop_21_adam_dense_201_kernel_m-
)assignvariableop_22_adam_dense_201_bias_m/
+assignvariableop_23_adam_dense_202_kernel_m-
)assignvariableop_24_adam_dense_202_bias_m/
+assignvariableop_25_adam_dense_203_kernel_m-
)assignvariableop_26_adam_dense_203_bias_m0
,assignvariableop_27_adam_conv2d_101_kernel_v.
*assignvariableop_28_adam_conv2d_101_bias_v/
+assignvariableop_29_adam_dense_201_kernel_v-
)assignvariableop_30_adam_dense_201_bias_v/
+assignvariableop_31_adam_dense_202_kernel_v-
)assignvariableop_32_adam_dense_202_bias_v/
+assignvariableop_33_adam_dense_203_kernel_v-
)assignvariableop_34_adam_dense_203_bias_v
identity_36¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9Ä
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*Ð
valueÆBÃ$B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB:layer-0/layer-0/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUEB:layer-0/layer-1/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesÖ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*[
valueRBP$B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesâ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*¦
_output_shapes
::::::::::::::::::::::::::::::::::::*2
dtypes(
&2$			2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity¡
AssignVariableOpAssignVariableOp"assignvariableop_conv2d_101_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1§
AssignVariableOp_1AssignVariableOp"assignvariableop_1_conv2d_101_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2¨
AssignVariableOp_2AssignVariableOp#assignvariableop_2_dense_201_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¦
AssignVariableOp_3AssignVariableOp!assignvariableop_3_dense_201_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4¨
AssignVariableOp_4AssignVariableOp#assignvariableop_4_dense_202_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5¦
AssignVariableOp_5AssignVariableOp!assignvariableop_5_dense_202_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6¨
AssignVariableOp_6AssignVariableOp#assignvariableop_6_dense_203_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7¦
AssignVariableOp_7AssignVariableOp!assignvariableop_7_dense_203_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_8¡
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9£
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_beta_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10§
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11¦
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12®
AssignVariableOp_12AssignVariableOp&assignvariableop_12_adam_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_13¤
AssignVariableOp_13AssignVariableOpassignvariableop_13_variableIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_14¦
AssignVariableOp_14AssignVariableOpassignvariableop_14_variable_1Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15¡
AssignVariableOp_15AssignVariableOpassignvariableop_15_totalIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16¡
AssignVariableOp_16AssignVariableOpassignvariableop_16_countIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17£
AssignVariableOp_17AssignVariableOpassignvariableop_17_total_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18£
AssignVariableOp_18AssignVariableOpassignvariableop_18_count_1Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19´
AssignVariableOp_19AssignVariableOp,assignvariableop_19_adam_conv2d_101_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20²
AssignVariableOp_20AssignVariableOp*assignvariableop_20_adam_conv2d_101_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21³
AssignVariableOp_21AssignVariableOp+assignvariableop_21_adam_dense_201_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22±
AssignVariableOp_22AssignVariableOp)assignvariableop_22_adam_dense_201_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23³
AssignVariableOp_23AssignVariableOp+assignvariableop_23_adam_dense_202_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24±
AssignVariableOp_24AssignVariableOp)assignvariableop_24_adam_dense_202_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25³
AssignVariableOp_25AssignVariableOp+assignvariableop_25_adam_dense_203_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26±
AssignVariableOp_26AssignVariableOp)assignvariableop_26_adam_dense_203_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27´
AssignVariableOp_27AssignVariableOp,assignvariableop_27_adam_conv2d_101_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28²
AssignVariableOp_28AssignVariableOp*assignvariableop_28_adam_conv2d_101_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29³
AssignVariableOp_29AssignVariableOp+assignvariableop_29_adam_dense_201_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30±
AssignVariableOp_30AssignVariableOp)assignvariableop_30_adam_dense_201_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31³
AssignVariableOp_31AssignVariableOp+assignvariableop_31_adam_dense_202_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32±
AssignVariableOp_32AssignVariableOp)assignvariableop_32_adam_dense_202_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33³
AssignVariableOp_33AssignVariableOp+assignvariableop_33_adam_dense_203_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34±
AssignVariableOp_34AssignVariableOp)assignvariableop_34_adam_dense_203_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_349
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpà
Identity_35Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_35Ó
Identity_36IdentityIdentity_35:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_36"#
identity_36Identity_36:output:0*£
_input_shapes
: :::::::::::::::::::::::::::::::::::2$
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
Á7
¡
"__inference__wrapped_model_1736498$
 training_data_augmentation_inputG
Csimplemodel_lessdense_avg_conv2d_101_conv2d_readvariableop_resourceH
Dsimplemodel_lessdense_avg_conv2d_101_biasadd_readvariableop_resourceF
Bsimplemodel_lessdense_avg_dense_201_matmul_readvariableop_resourceG
Csimplemodel_lessdense_avg_dense_201_biasadd_readvariableop_resourceF
Bsimplemodel_lessdense_avg_dense_202_matmul_readvariableop_resourceG
Csimplemodel_lessdense_avg_dense_202_biasadd_readvariableop_resourceF
Bsimplemodel_lessdense_avg_dense_203_matmul_readvariableop_resourceG
Csimplemodel_lessdense_avg_dense_203_biasadd_readvariableop_resource
identity
:SimpleModel_LessDense_avg/conv2d_101/Conv2D/ReadVariableOpReadVariableOpCsimplemodel_lessdense_avg_conv2d_101_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02<
:SimpleModel_LessDense_avg/conv2d_101/Conv2D/ReadVariableOp­
+SimpleModel_LessDense_avg/conv2d_101/Conv2DConv2D training_data_augmentation_inputBSimpleModel_LessDense_avg/conv2d_101/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2-
+SimpleModel_LessDense_avg/conv2d_101/Conv2Dû
;SimpleModel_LessDense_avg/conv2d_101/BiasAdd/ReadVariableOpReadVariableOpDsimplemodel_lessdense_avg_conv2d_101_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02=
;SimpleModel_LessDense_avg/conv2d_101/BiasAdd/ReadVariableOp
,SimpleModel_LessDense_avg/conv2d_101/BiasAddBiasAdd4SimpleModel_LessDense_avg/conv2d_101/Conv2D:output:0CSimpleModel_LessDense_avg/conv2d_101/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2.
,SimpleModel_LessDense_avg/conv2d_101/BiasAddÏ
)SimpleModel_LessDense_avg/conv2d_101/ReluRelu5SimpleModel_LessDense_avg/conv2d_101/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2+
)SimpleModel_LessDense_avg/conv2d_101/Reluª
6SimpleModel_LessDense_avg/average_pooling2d_46/AvgPoolAvgPool7SimpleModel_LessDense_avg/conv2d_101/Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
28
6SimpleModel_LessDense_avg/average_pooling2d_46/AvgPoolç
.SimpleModel_LessDense_avg/dropout_101/IdentityIdentity?SimpleModel_LessDense_avg/average_pooling2d_46/AvgPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@20
.SimpleModel_LessDense_avg/dropout_101/Identity©
*SimpleModel_LessDense_avg/flatten_67/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@*  2,
*SimpleModel_LessDense_avg/flatten_67/Const
,SimpleModel_LessDense_avg/flatten_67/ReshapeReshape7SimpleModel_LessDense_avg/dropout_101/Identity:output:03SimpleModel_LessDense_avg/flatten_67/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀT2.
,SimpleModel_LessDense_avg/flatten_67/Reshapeû
9SimpleModel_LessDense_avg/dense_201/MatMul/ReadVariableOpReadVariableOpBsimplemodel_lessdense_avg_dense_201_matmul_readvariableop_resource* 
_output_shapes
:
ÀT*
dtype02;
9SimpleModel_LessDense_avg/dense_201/MatMul/ReadVariableOp
*SimpleModel_LessDense_avg/dense_201/MatMulMatMul5SimpleModel_LessDense_avg/flatten_67/Reshape:output:0ASimpleModel_LessDense_avg/dense_201/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*SimpleModel_LessDense_avg/dense_201/MatMulù
:SimpleModel_LessDense_avg/dense_201/BiasAdd/ReadVariableOpReadVariableOpCsimplemodel_lessdense_avg_dense_201_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02<
:SimpleModel_LessDense_avg/dense_201/BiasAdd/ReadVariableOp
+SimpleModel_LessDense_avg/dense_201/BiasAddBiasAdd4SimpleModel_LessDense_avg/dense_201/MatMul:product:0BSimpleModel_LessDense_avg/dense_201/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+SimpleModel_LessDense_avg/dense_201/BiasAddÅ
(SimpleModel_LessDense_avg/dense_201/ReluRelu4SimpleModel_LessDense_avg/dense_201/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(SimpleModel_LessDense_avg/dense_201/Reluú
9SimpleModel_LessDense_avg/dense_202/MatMul/ReadVariableOpReadVariableOpBsimplemodel_lessdense_avg_dense_202_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02;
9SimpleModel_LessDense_avg/dense_202/MatMul/ReadVariableOp
*SimpleModel_LessDense_avg/dense_202/MatMulMatMul6SimpleModel_LessDense_avg/dense_201/Relu:activations:0ASimpleModel_LessDense_avg/dense_202/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2,
*SimpleModel_LessDense_avg/dense_202/MatMulø
:SimpleModel_LessDense_avg/dense_202/BiasAdd/ReadVariableOpReadVariableOpCsimplemodel_lessdense_avg_dense_202_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02<
:SimpleModel_LessDense_avg/dense_202/BiasAdd/ReadVariableOp
+SimpleModel_LessDense_avg/dense_202/BiasAddBiasAdd4SimpleModel_LessDense_avg/dense_202/MatMul:product:0BSimpleModel_LessDense_avg/dense_202/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2-
+SimpleModel_LessDense_avg/dense_202/BiasAddÄ
(SimpleModel_LessDense_avg/dense_202/ReluRelu4SimpleModel_LessDense_avg/dense_202/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2*
(SimpleModel_LessDense_avg/dense_202/Reluù
9SimpleModel_LessDense_avg/dense_203/MatMul/ReadVariableOpReadVariableOpBsimplemodel_lessdense_avg_dense_203_matmul_readvariableop_resource*
_output_shapes

:@
*
dtype02;
9SimpleModel_LessDense_avg/dense_203/MatMul/ReadVariableOp
*SimpleModel_LessDense_avg/dense_203/MatMulMatMul6SimpleModel_LessDense_avg/dense_202/Relu:activations:0ASimpleModel_LessDense_avg/dense_203/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2,
*SimpleModel_LessDense_avg/dense_203/MatMulø
:SimpleModel_LessDense_avg/dense_203/BiasAdd/ReadVariableOpReadVariableOpCsimplemodel_lessdense_avg_dense_203_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02<
:SimpleModel_LessDense_avg/dense_203/BiasAdd/ReadVariableOp
+SimpleModel_LessDense_avg/dense_203/BiasAddBiasAdd4SimpleModel_LessDense_avg/dense_203/MatMul:product:0BSimpleModel_LessDense_avg/dense_203/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2-
+SimpleModel_LessDense_avg/dense_203/BiasAddÍ
+SimpleModel_LessDense_avg/dense_203/SoftmaxSoftmax4SimpleModel_LessDense_avg/dense_203/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2-
+SimpleModel_LessDense_avg/dense_203/Softmax
IdentityIdentity5SimpleModel_LessDense_avg/dense_203/Softmax:softmax:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ:::::::::q m
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:
_user_specified_name" Training_Data_Augmentation_input
	

;__inference_SimpleModel_LessDense_avg_layer_call_fn_1737179$
 training_data_augmentation_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity¢StatefulPartitionedCallî
StatefulPartitionedCallStatefulPartitionedCall training_data_augmentation_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *_
fZRX
V__inference_SimpleModel_LessDense_avg_layer_call_and_return_conditional_losses_17371602
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:q m
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:
_user_specified_name" Training_Data_Augmentation_input
ã

+__inference_dense_202_layer_call_fn_1737721

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallö
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_202_layer_call_and_return_conditional_losses_17370042
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
²
½
L__inference_random_rotation_layer_call_and_return_conditional_losses_1736775

inputs-
)stateful_uniform_statefuluniform_resource
identity¢ stateful_uniform/StatefulUniformD
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
strided_slice/stack_2â
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
strided_slice_1/stack_2ì
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
strided_slice_2/stack_2ì
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
 *|Ù ¿2
stateful_uniform/minq
stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *|Ù ?2
stateful_uniform/max
*stateful_uniform/StatefulUniform/algorithmConst*
_output_shapes
: *
dtype0	*
value	B	 R2,
*stateful_uniform/StatefulUniform/algorithm
 stateful_uniform/StatefulUniformStatefulUniform)stateful_uniform_statefuluniform_resource3stateful_uniform/StatefulUniform/algorithm:output:0stateful_uniform/shape:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
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
:ÿÿÿÿÿÿÿÿÿ2
stateful_uniform/mul
stateful_uniformAddstateful_uniform/mul:z:0stateful_uniform/min:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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
:ÿÿÿÿÿÿÿÿÿ2
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
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/mulu
rotation_matrix/SinSinstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/mul_1
rotation_matrix/sub_3Subrotation_matrix/mul:z:0rotation_matrix/mul_1:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/sub_3
rotation_matrix/sub_4Subrotation_matrix/sub:z:0rotation_matrix/sub_3:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/sub_4{
rotation_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
rotation_matrix/truediv/yª
rotation_matrix/truedivRealDivrotation_matrix/sub_4:z:0"rotation_matrix/truediv/y:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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
:ÿÿÿÿÿÿÿÿÿ2
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
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/mul_2y
rotation_matrix/Cos_1Cosstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/mul_3
rotation_matrix/addAddV2rotation_matrix/mul_2:z:0rotation_matrix/mul_3:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/add
rotation_matrix/sub_8Subrotation_matrix/sub_5:z:0rotation_matrix/add:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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
:ÿÿÿÿÿÿÿÿÿ2
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
%rotation_matrix/strided_slice/stack_2Â
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
:ÿÿÿÿÿÿÿÿÿ2
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
'rotation_matrix/strided_slice_1/stack_2÷
rotation_matrix/strided_slice_1StridedSlicerotation_matrix/Cos_2:y:0.rotation_matrix/strided_slice_1/stack:output:00rotation_matrix/strided_slice_1/stack_1:output:00rotation_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_1y
rotation_matrix/Sin_2Sinstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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
'rotation_matrix/strided_slice_2/stack_2÷
rotation_matrix/strided_slice_2StridedSlicerotation_matrix/Sin_2:y:0.rotation_matrix/strided_slice_2/stack:output:00rotation_matrix/strided_slice_2/stack_1:output:00rotation_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_2
rotation_matrix/NegNeg(rotation_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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
'rotation_matrix/strided_slice_3/stack_2ù
rotation_matrix/strided_slice_3StridedSlicerotation_matrix/truediv:z:0.rotation_matrix/strided_slice_3/stack:output:00rotation_matrix/strided_slice_3/stack_1:output:00rotation_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_3y
rotation_matrix/Sin_3Sinstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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
'rotation_matrix/strided_slice_4/stack_2÷
rotation_matrix/strided_slice_4StridedSlicerotation_matrix/Sin_3:y:0.rotation_matrix/strided_slice_4/stack:output:00rotation_matrix/strided_slice_4/stack_1:output:00rotation_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_4y
rotation_matrix/Cos_3Cosstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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
'rotation_matrix/strided_slice_5/stack_2÷
rotation_matrix/strided_slice_5StridedSlicerotation_matrix/Cos_3:y:0.rotation_matrix/strided_slice_5/stack:output:00rotation_matrix/strided_slice_5/stack_1:output:00rotation_matrix/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

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
'rotation_matrix/strided_slice_6/stack_2û
rotation_matrix/strided_slice_6StridedSlicerotation_matrix/truediv_1:z:0.rotation_matrix/strided_slice_6/stack:output:00rotation_matrix/strided_slice_6/stack_1:output:00rotation_matrix/strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

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
B :è2
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
rotation_matrix/zeros/packed/1Ã
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
rotation_matrix/zeros/Constµ
rotation_matrix/zerosFill%rotation_matrix/zeros/packed:output:0$rotation_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
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
:ÿÿÿÿÿÿÿÿÿ2
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
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR2&
$transform/ImageProjectiveTransformV2¸
IdentityIdentity9transform/ImageProjectiveTransformV2:transformed_images:0!^stateful_uniform/StatefulUniform*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ:2D
 stateful_uniform/StatefulUniform stateful_uniform/StatefulUniform:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
¯
G__inference_conv2d_101_layer_call_and_return_conditional_losses_1737634

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
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
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ:::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ë
f
H__inference_dropout_101_layer_call_and_return_conditional_losses_1737660

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Æ
f
-__inference_random_flip_layer_call_fn_1737778

inputs
identity¢StatefulPartitionedCallæ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_random_flip_layer_call_and_return_conditional_losses_17366532
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
á

+__inference_dense_203_layer_call_fn_1737741

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallö
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_203_layer_call_and_return_conditional_losses_17370312
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Â
M
1__inference_random_contrast_layer_call_fn_1737998

inputs
identityÒ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_random_contrast_layer_call_and_return_conditional_losses_17368132
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


<__inference_Training_Data_Augmentation_layer_call_fn_1737618

inputs
unknown
identity¢StatefulPartitionedCallÿ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *`
f[RY
W__inference_Training_Data_Augmentation_layer_call_and_return_conditional_losses_17368462
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ:22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ö
h
L__inference_random_rotation_layer_call_and_return_conditional_losses_1736779

inputs
identityb
IdentityIdentityinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
º
I
-__inference_random_flip_layer_call_fn_1737783

inputs
identityÎ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_random_flip_layer_call_and_return_conditional_losses_17366572
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ª
H
,__inference_flatten_67_layer_call_fn_1737681

inputs
identityÆ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀT* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_67_layer_call_and_return_conditional_losses_17369582
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀT2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
õ$
Ú
V__inference_SimpleModel_LessDense_avg_layer_call_and_return_conditional_losses_1737076$
 training_data_augmentation_input
conv2d_101_1737052
conv2d_101_1737054
dense_201_1737060
dense_201_1737062
dense_202_1737065
dense_202_1737067
dense_203_1737070
dense_203_1737072
identity¢"conv2d_101/StatefulPartitionedCall¢!dense_201/StatefulPartitionedCall¢!dense_202/StatefulPartitionedCall¢!dense_203/StatefulPartitionedCall­
*Training_Data_Augmentation/PartitionedCallPartitionedCall training_data_augmentation_input*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *`
f[RY
W__inference_Training_Data_Augmentation_layer_call_and_return_conditional_losses_17368602,
*Training_Data_Augmentation/PartitionedCallÖ
"conv2d_101/StatefulPartitionedCallStatefulPartitionedCall3Training_Data_Augmentation/PartitionedCall:output:0conv2d_101_1737052conv2d_101_1737054*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_conv2d_101_layer_call_and_return_conditional_losses_17369052$
"conv2d_101/StatefulPartitionedCall¦
$average_pooling2d_46/PartitionedCallPartitionedCall+conv2d_101/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_average_pooling2d_46_layer_call_and_return_conditional_losses_17368692&
$average_pooling2d_46/PartitionedCall
dropout_101/PartitionedCallPartitionedCall-average_pooling2d_46/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_101_layer_call_and_return_conditional_losses_17369392
dropout_101/PartitionedCallú
flatten_67/PartitionedCallPartitionedCall$dropout_101/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀT* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_67_layer_call_and_return_conditional_losses_17369582
flatten_67/PartitionedCallº
!dense_201/StatefulPartitionedCallStatefulPartitionedCall#flatten_67/PartitionedCall:output:0dense_201_1737060dense_201_1737062*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_201_layer_call_and_return_conditional_losses_17369772#
!dense_201/StatefulPartitionedCallÀ
!dense_202/StatefulPartitionedCallStatefulPartitionedCall*dense_201/StatefulPartitionedCall:output:0dense_202_1737065dense_202_1737067*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_202_layer_call_and_return_conditional_losses_17370042#
!dense_202/StatefulPartitionedCallÀ
!dense_203/StatefulPartitionedCallStatefulPartitionedCall*dense_202/StatefulPartitionedCall:output:0dense_203_1737070dense_203_1737072*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_203_layer_call_and_return_conditional_losses_17370312#
!dense_203/StatefulPartitionedCall
IdentityIdentity*dense_203/StatefulPartitionedCall:output:0#^conv2d_101/StatefulPartitionedCall"^dense_201/StatefulPartitionedCall"^dense_202/StatefulPartitionedCall"^dense_203/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ::::::::2H
"conv2d_101/StatefulPartitionedCall"conv2d_101/StatefulPartitionedCall2F
!dense_201/StatefulPartitionedCall!dense_201/StatefulPartitionedCall2F
!dense_202/StatefulPartitionedCall!dense_202/StatefulPartitionedCall2F
!dense_203/StatefulPartitionedCall!dense_203/StatefulPartitionedCall:q m
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:
_user_specified_name" Training_Data_Augmentation_input
Ã
d
H__inference_random_flip_layer_call_and_return_conditional_losses_1736619

inputs
identity}
IdentityIdentityinputs*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
²%
ë
V__inference_SimpleModel_LessDense_avg_layer_call_and_return_conditional_losses_1737424

inputs-
)conv2d_101_conv2d_readvariableop_resource.
*conv2d_101_biasadd_readvariableop_resource,
(dense_201_matmul_readvariableop_resource-
)dense_201_biasadd_readvariableop_resource,
(dense_202_matmul_readvariableop_resource-
)dense_202_biasadd_readvariableop_resource,
(dense_203_matmul_readvariableop_resource-
)dense_203_biasadd_readvariableop_resource
identity¶
 conv2d_101/Conv2D/ReadVariableOpReadVariableOp)conv2d_101_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02"
 conv2d_101/Conv2D/ReadVariableOpÅ
conv2d_101/Conv2DConv2Dinputs(conv2d_101/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
conv2d_101/Conv2D­
!conv2d_101/BiasAdd/ReadVariableOpReadVariableOp*conv2d_101_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02#
!conv2d_101/BiasAdd/ReadVariableOp´
conv2d_101/BiasAddBiasAddconv2d_101/Conv2D:output:0)conv2d_101/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d_101/BiasAdd
conv2d_101/ReluReluconv2d_101/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d_101/ReluÜ
average_pooling2d_46/AvgPoolAvgPoolconv2d_101/Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
2
average_pooling2d_46/AvgPool
dropout_101/IdentityIdentity%average_pooling2d_46/AvgPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_101/Identityu
flatten_67/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@*  2
flatten_67/Const 
flatten_67/ReshapeReshapedropout_101/Identity:output:0flatten_67/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀT2
flatten_67/Reshape­
dense_201/MatMul/ReadVariableOpReadVariableOp(dense_201_matmul_readvariableop_resource* 
_output_shapes
:
ÀT*
dtype02!
dense_201/MatMul/ReadVariableOp§
dense_201/MatMulMatMulflatten_67/Reshape:output:0'dense_201/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_201/MatMul«
 dense_201/BiasAdd/ReadVariableOpReadVariableOp)dense_201_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02"
 dense_201/BiasAdd/ReadVariableOpª
dense_201/BiasAddBiasAdddense_201/MatMul:product:0(dense_201/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_201/BiasAddw
dense_201/ReluReludense_201/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_201/Relu¬
dense_202/MatMul/ReadVariableOpReadVariableOp(dense_202_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02!
dense_202/MatMul/ReadVariableOp§
dense_202/MatMulMatMuldense_201/Relu:activations:0'dense_202/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_202/MatMulª
 dense_202/BiasAdd/ReadVariableOpReadVariableOp)dense_202_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 dense_202/BiasAdd/ReadVariableOp©
dense_202/BiasAddBiasAdddense_202/MatMul:product:0(dense_202/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_202/BiasAddv
dense_202/ReluReludense_202/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_202/Relu«
dense_203/MatMul/ReadVariableOpReadVariableOp(dense_203_matmul_readvariableop_resource*
_output_shapes

:@
*
dtype02!
dense_203/MatMul/ReadVariableOp§
dense_203/MatMulMatMuldense_202/Relu:activations:0'dense_203/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_203/MatMulª
 dense_203/BiasAdd/ReadVariableOpReadVariableOp)dense_203_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02"
 dense_203/BiasAdd/ReadVariableOp©
dense_203/BiasAddBiasAdddense_203/MatMul:product:0(dense_203/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_203/BiasAdd
dense_203/SoftmaxSoftmaxdense_203/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_203/Softmaxo
IdentityIdentitydense_203/Softmax:softmax:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ:::::::::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ö
h
L__inference_random_rotation_layer_call_and_return_conditional_losses_1737959

inputs
identityb
IdentityIdentityinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¡

<__inference_Training_Data_Augmentation_layer_call_fn_1736851
random_flip_input
unknown
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallrandom_flip_inputunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *`
f[RY
W__inference_Training_Data_Augmentation_layer_call_and_return_conditional_losses_17368462
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ:22
StatefulPartitionedCallStatefulPartitionedCall:b ^
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+
_user_specified_namerandom_flip_input
Ø
~
W__inference_Training_Data_Augmentation_layer_call_and_return_conditional_losses_1736834
random_flip_input
identityñ
random_flip/PartitionedCallPartitionedCallrandom_flip_input*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_random_flip_layer_call_and_return_conditional_losses_17366572
random_flip/PartitionedCall
random_rotation/PartitionedCallPartitionedCall$random_flip/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_random_rotation_layer_call_and_return_conditional_losses_17367792!
random_rotation/PartitionedCall
random_contrast/PartitionedCallPartitionedCall(random_rotation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_random_contrast_layer_call_and_return_conditional_losses_17368132!
random_contrast/PartitionedCall
IdentityIdentity(random_contrast/PartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:b ^
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+
_user_specified_namerandom_flip_input
å

+__inference_dense_201_layer_call_fn_1737701

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall÷
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_201_layer_call_and_return_conditional_losses_17369772
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿÀT::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀT
 
_user_specified_nameinputs
ÝK
³
 __inference__traced_save_1738136
file_prefix0
,savev2_conv2d_101_kernel_read_readvariableop.
*savev2_conv2d_101_bias_read_readvariableop/
+savev2_dense_201_kernel_read_readvariableop-
)savev2_dense_201_bias_read_readvariableop/
+savev2_dense_202_kernel_read_readvariableop-
)savev2_dense_202_bias_read_readvariableop/
+savev2_dense_203_kernel_read_readvariableop-
)savev2_dense_203_bias_read_readvariableop(
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
3savev2_adam_conv2d_101_kernel_m_read_readvariableop5
1savev2_adam_conv2d_101_bias_m_read_readvariableop6
2savev2_adam_dense_201_kernel_m_read_readvariableop4
0savev2_adam_dense_201_bias_m_read_readvariableop6
2savev2_adam_dense_202_kernel_m_read_readvariableop4
0savev2_adam_dense_202_bias_m_read_readvariableop6
2savev2_adam_dense_203_kernel_m_read_readvariableop4
0savev2_adam_dense_203_bias_m_read_readvariableop7
3savev2_adam_conv2d_101_kernel_v_read_readvariableop5
1savev2_adam_conv2d_101_bias_v_read_readvariableop6
2savev2_adam_dense_201_kernel_v_read_readvariableop4
0savev2_adam_dense_201_bias_v_read_readvariableop6
2savev2_adam_dense_202_kernel_v_read_readvariableop4
0savev2_adam_dense_202_bias_v_read_readvariableop6
2savev2_adam_dense_203_kernel_v_read_readvariableop4
0savev2_adam_dense_203_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
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
value3B1 B+_temp_26112031d9d445f98c03674d18422653/part2	
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
ShardedFilename¾
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*Ð
valueÆBÃ$B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB:layer-0/layer-0/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUEB:layer-0/layer-1/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesÐ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*[
valueRBP$B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0,savev2_conv2d_101_kernel_read_readvariableop*savev2_conv2d_101_bias_read_readvariableop+savev2_dense_201_kernel_read_readvariableop)savev2_dense_201_bias_read_readvariableop+savev2_dense_202_kernel_read_readvariableop)savev2_dense_202_bias_read_readvariableop+savev2_dense_203_kernel_read_readvariableop)savev2_dense_203_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop#savev2_variable_read_readvariableop%savev2_variable_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop3savev2_adam_conv2d_101_kernel_m_read_readvariableop1savev2_adam_conv2d_101_bias_m_read_readvariableop2savev2_adam_dense_201_kernel_m_read_readvariableop0savev2_adam_dense_201_bias_m_read_readvariableop2savev2_adam_dense_202_kernel_m_read_readvariableop0savev2_adam_dense_202_bias_m_read_readvariableop2savev2_adam_dense_203_kernel_m_read_readvariableop0savev2_adam_dense_203_bias_m_read_readvariableop3savev2_adam_conv2d_101_kernel_v_read_readvariableop1savev2_adam_conv2d_101_bias_v_read_readvariableop2savev2_adam_dense_201_kernel_v_read_readvariableop0savev2_adam_dense_201_bias_v_read_readvariableop2savev2_adam_dense_202_kernel_v_read_readvariableop0savev2_adam_dense_202_bias_v_read_readvariableop2savev2_adam_dense_203_kernel_v_read_readvariableop0savev2_adam_dense_203_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *2
dtypes(
&2$			2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
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

identity_1Identity_1:output:0*
_input_shapes
: :@:@:
ÀT::	@:@:@
:
: : : : : ::: : : : :@:@:
ÀT::	@:@:@
:
:@:@:
ÀT::	@:@:@
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
:@:&"
 
_output_shapes
:
ÀT:!

_output_shapes	
::%!

_output_shapes
:	@: 

_output_shapes
:@:$ 

_output_shapes

:@
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
:@: 

_output_shapes
:@:&"
 
_output_shapes
:
ÀT:!

_output_shapes	
::%!

_output_shapes
:	@: 

_output_shapes
:@:$ 

_output_shapes

:@
: 

_output_shapes
:
:,(
&
_output_shapes
:@: 

_output_shapes
:@:&"
 
_output_shapes
:
ÀT:!

_output_shapes	
::% !

_output_shapes
:	@: !

_output_shapes
:@:$" 

_output_shapes

:@
: #

_output_shapes
:
:$

_output_shapes
: 
¿
c
G__inference_flatten_67_layer_call_and_return_conditional_losses_1736958

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ@*  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀT2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀT2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ö
h
L__inference_random_contrast_layer_call_and_return_conditional_losses_1737988

inputs
identityb
IdentityIdentityinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Õ
î
%__inference_signature_wrapper_1737210$
 training_data_augmentation_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity¢StatefulPartitionedCallº
StatefulPartitionedCallStatefulPartitionedCall training_data_augmentation_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__wrapped_model_17364982
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:q m
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:
_user_specified_name" Training_Data_Augmentation_input
Æ
f
-__inference_dropout_101_layer_call_fn_1737665

inputs
identity¢StatefulPartitionedCallæ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_101_layer_call_and_return_conditional_losses_17369342
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ò(
Ý
V__inference_SimpleModel_LessDense_avg_layer_call_and_return_conditional_losses_1737048$
 training_data_augmentation_input&
"training_data_augmentation_1736891
conv2d_101_1736916
conv2d_101_1736918
dense_201_1736988
dense_201_1736990
dense_202_1737015
dense_202_1737017
dense_203_1737042
dense_203_1737044
identity¢2Training_Data_Augmentation/StatefulPartitionedCall¢"conv2d_101/StatefulPartitionedCall¢!dense_201/StatefulPartitionedCall¢!dense_202/StatefulPartitionedCall¢!dense_203/StatefulPartitionedCall¢#dropout_101/StatefulPartitionedCallê
2Training_Data_Augmentation/StatefulPartitionedCallStatefulPartitionedCall training_data_augmentation_input"training_data_augmentation_1736891*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *`
f[RY
W__inference_Training_Data_Augmentation_layer_call_and_return_conditional_losses_173684624
2Training_Data_Augmentation/StatefulPartitionedCallÞ
"conv2d_101/StatefulPartitionedCallStatefulPartitionedCall;Training_Data_Augmentation/StatefulPartitionedCall:output:0conv2d_101_1736916conv2d_101_1736918*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_conv2d_101_layer_call_and_return_conditional_losses_17369052$
"conv2d_101/StatefulPartitionedCall¦
$average_pooling2d_46/PartitionedCallPartitionedCall+conv2d_101/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Z
fURS
Q__inference_average_pooling2d_46_layer_call_and_return_conditional_losses_17368692&
$average_pooling2d_46/PartitionedCall¥
#dropout_101/StatefulPartitionedCallStatefulPartitionedCall-average_pooling2d_46/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_101_layer_call_and_return_conditional_losses_17369342%
#dropout_101/StatefulPartitionedCall
flatten_67/PartitionedCallPartitionedCall,dropout_101/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀT* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_67_layer_call_and_return_conditional_losses_17369582
flatten_67/PartitionedCallº
!dense_201/StatefulPartitionedCallStatefulPartitionedCall#flatten_67/PartitionedCall:output:0dense_201_1736988dense_201_1736990*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_201_layer_call_and_return_conditional_losses_17369772#
!dense_201/StatefulPartitionedCallÀ
!dense_202/StatefulPartitionedCallStatefulPartitionedCall*dense_201/StatefulPartitionedCall:output:0dense_202_1737015dense_202_1737017*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_202_layer_call_and_return_conditional_losses_17370042#
!dense_202/StatefulPartitionedCallÀ
!dense_203/StatefulPartitionedCallStatefulPartitionedCall*dense_202/StatefulPartitionedCall:output:0dense_203_1737042dense_203_1737044*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dense_203_layer_call_and_return_conditional_losses_17370312#
!dense_203/StatefulPartitionedCallê
IdentityIdentity*dense_203/StatefulPartitionedCall:output:03^Training_Data_Augmentation/StatefulPartitionedCall#^conv2d_101/StatefulPartitionedCall"^dense_201/StatefulPartitionedCall"^dense_202/StatefulPartitionedCall"^dense_203/StatefulPartitionedCall$^dropout_101/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ:::::::::2h
2Training_Data_Augmentation/StatefulPartitionedCall2Training_Data_Augmentation/StatefulPartitionedCall2H
"conv2d_101/StatefulPartitionedCall"conv2d_101/StatefulPartitionedCall2F
!dense_201/StatefulPartitionedCall!dense_201/StatefulPartitionedCall2F
!dense_202/StatefulPartitionedCall!dense_202/StatefulPartitionedCall2F
!dense_203/StatefulPartitionedCall!dense_203/StatefulPartitionedCall2J
#dropout_101/StatefulPartitionedCall#dropout_101/StatefulPartitionedCall:q m
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:
_user_specified_name" Training_Data_Augmentation_input
®
®
F__inference_dense_202_layer_call_and_return_conditional_losses_1737712

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
:ÿÿÿÿÿÿÿÿÿ@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Reluf
IdentityIdentityRelu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Æ
g
H__inference_dropout_101_layer_call_and_return_conditional_losses_1736934

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¼
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2
dropout/GreaterEqual/yÆ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

m
Q__inference_average_pooling2d_46_layer_call_and_return_conditional_losses_1736869

inputs
identity¶
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
AvgPool
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ñ
ê
;__inference_SimpleModel_LessDense_avg_layer_call_fn_1737468

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity¢StatefulPartitionedCallÔ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *_
fZRX
V__inference_SimpleModel_LessDense_avg_layer_call_and_return_conditional_losses_17371602
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿ::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs"¸L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*æ
serving_defaultÒ
u
 Training_Data_Augmentation_inputQ
2serving_default_Training_Data_Augmentation_input:0ÿÿÿÿÿÿÿÿÿ=
	dense_2030
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿ
tensorflow/serving/predict:è
èD
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

regularization_losses
	variables
trainable_variables
	keras_api

signatures
¦__call__
§_default_save_signature
+¨&call_and_return_all_conditional_losses"ÊA
_tf_keras_sequential«A{"class_name": "Sequential", "name": "SimpleModel_LessDense_avg", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "SimpleModel_LessDense_avg", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Training_Data_Augmentation_input"}}, {"class_name": "Sequential", "config": {"name": "Training_Data_Augmentation", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "random_flip_input"}}, {"class_name": "RandomFlip", "config": {"name": "random_flip", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "mode": "horizontal", "seed": null}}, {"class_name": "RandomRotation", "config": {"name": "random_rotation", "trainable": true, "dtype": "float32", "factor": 0.2, "fill_mode": "reflect", "interpolation": "bilinear", "seed": null}}, {"class_name": "RandomContrast", "config": {"name": "random_contrast", "trainable": true, "dtype": "float32", "factor": 0.1, "seed": null}}]}}, {"class_name": "Conv2D", "config": {"name": "conv2d_101", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "AveragePooling2D", "config": {"name": "average_pooling2d_46", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_101", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten_67", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_201", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_202", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_203", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "SimpleModel_LessDense_avg", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Training_Data_Augmentation_input"}}, {"class_name": "Sequential", "config": {"name": "Training_Data_Augmentation", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "random_flip_input"}}, {"class_name": "RandomFlip", "config": {"name": "random_flip", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "mode": "horizontal", "seed": null}}, {"class_name": "RandomRotation", "config": {"name": "random_rotation", "trainable": true, "dtype": "float32", "factor": 0.2, "fill_mode": "reflect", "interpolation": "bilinear", "seed": null}}, {"class_name": "RandomContrast", "config": {"name": "random_contrast", "trainable": true, "dtype": "float32", "factor": 0.1, "seed": null}}]}}, {"class_name": "Conv2D", "config": {"name": "conv2d_101", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "AveragePooling2D", "config": {"name": "average_pooling2d_46", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_101", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten_67", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_201", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_202", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_203", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "sparse_categorical_crossentropy", "metrics": ["accuracy"], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.009999999776482582, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
Û
layer-0
layer-1
layer-2
regularization_losses
	variables
trainable_variables
	keras_api
©__call__
+ª&call_and_return_all_conditional_losses"£
_tf_keras_sequential{"class_name": "Sequential", "name": "Training_Data_Augmentation", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "Training_Data_Augmentation", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "random_flip_input"}}, {"class_name": "RandomFlip", "config": {"name": "random_flip", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "mode": "horizontal", "seed": null}}, {"class_name": "RandomRotation", "config": {"name": "random_rotation", "trainable": true, "dtype": "float32", "factor": 0.2, "fill_mode": "reflect", "interpolation": "bilinear", "seed": null}}, {"class_name": "RandomContrast", "config": {"name": "random_contrast", "trainable": true, "dtype": "float32", "factor": 0.1, "seed": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "Training_Data_Augmentation", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "random_flip_input"}}, {"class_name": "RandomFlip", "config": {"name": "random_flip", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "mode": "horizontal", "seed": null}}, {"class_name": "RandomRotation", "config": {"name": "random_rotation", "trainable": true, "dtype": "float32", "factor": 0.2, "fill_mode": "reflect", "interpolation": "bilinear", "seed": null}}, {"class_name": "RandomContrast", "config": {"name": "random_contrast", "trainable": true, "dtype": "float32", "factor": 0.1, "seed": null}}]}}}
ø


kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
«__call__
+¬&call_and_return_all_conditional_losses"Ñ	
_tf_keras_layer·	{"class_name": "Conv2D", "name": "conv2d_101", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_101", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 1]}}

regularization_losses
	variables
trainable_variables
	keras_api
­__call__
+®&call_and_return_all_conditional_losses"þ
_tf_keras_layerä{"class_name": "AveragePooling2D", "name": "average_pooling2d_46", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "average_pooling2d_46", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ë
 regularization_losses
!	variables
"trainable_variables
#	keras_api
¯__call__
+°&call_and_return_all_conditional_losses"Ú
_tf_keras_layerÀ{"class_name": "Dropout", "name": "dropout_101", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_101", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}}
ê
$regularization_losses
%	variables
&trainable_variables
'	keras_api
±__call__
+²&call_and_return_all_conditional_losses"Ù
_tf_keras_layer¿{"class_name": "Flatten", "name": "flatten_67", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_67", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
ý

(kernel
)bias
*regularization_losses
+	variables
,trainable_variables
-	keras_api
³__call__
+´&call_and_return_all_conditional_losses"Ö
_tf_keras_layer¼{"class_name": "Dense", "name": "dense_201", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_201", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 10816}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10816]}}
ø

.kernel
/bias
0regularization_losses
1	variables
2trainable_variables
3	keras_api
µ__call__
+¶&call_and_return_all_conditional_losses"Ñ
_tf_keras_layer·{"class_name": "Dense", "name": "dense_202", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_202", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
ù

4kernel
5bias
6regularization_losses
7	variables
8trainable_variables
9	keras_api
·__call__
+¸&call_and_return_all_conditional_losses"Ò
_tf_keras_layer¸{"class_name": "Dense", "name": "dense_203", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_203", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
ó
:iter

;beta_1

<beta_2
	=decay
>learning_ratemm(m)m.m/m4m5mvv(v )v¡.v¢/v£4v¤5v¥"
	optimizer
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
Î

?layers
@layer_metrics
Anon_trainable_variables

regularization_losses
	variables
trainable_variables
Bmetrics
Clayer_regularization_losses
¦__call__
§_default_save_signature
+¨&call_and_return_all_conditional_losses
'¨"call_and_return_conditional_losses"
_generic_user_object
-
¹serving_default"
signature_map
ý
D_rng
Eregularization_losses
F	variables
Gtrainable_variables
H	keras_api
º__call__
+»&call_and_return_all_conditional_losses"â
_tf_keras_layerÈ{"class_name": "RandomFlip", "name": "random_flip", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "random_flip", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "mode": "horizontal", "seed": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
¦
I_rng
Jregularization_losses
K	variables
Ltrainable_variables
M	keras_api
¼__call__
+½&call_and_return_all_conditional_losses"
_tf_keras_layerñ{"class_name": "RandomRotation", "name": "random_rotation", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "random_rotation", "trainable": true, "dtype": "float32", "factor": 0.2, "fill_mode": "reflect", "interpolation": "bilinear", "seed": null}}
ç
Nregularization_losses
O	variables
Ptrainable_variables
Q	keras_api
¾__call__
+¿&call_and_return_all_conditional_losses"Ö
_tf_keras_layer¼{"class_name": "RandomContrast", "name": "random_contrast", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "random_contrast", "trainable": true, "dtype": "float32", "factor": 0.1, "seed": null}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°

Rlayers
Slayer_metrics
Tnon_trainable_variables
regularization_losses
	variables
trainable_variables
Umetrics
Vlayer_regularization_losses
©__call__
+ª&call_and_return_all_conditional_losses
'ª"call_and_return_conditional_losses"
_generic_user_object
+:)@2conv2d_101/kernel
:@2conv2d_101/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
°

Wlayers
Xlayer_metrics
Ynon_trainable_variables
regularization_losses
	variables
trainable_variables
Zmetrics
[layer_regularization_losses
«__call__
+¬&call_and_return_all_conditional_losses
'¬"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°

\layers
]layer_metrics
^non_trainable_variables
regularization_losses
	variables
trainable_variables
_metrics
`layer_regularization_losses
­__call__
+®&call_and_return_all_conditional_losses
'®"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°

alayers
blayer_metrics
cnon_trainable_variables
 regularization_losses
!	variables
"trainable_variables
dmetrics
elayer_regularization_losses
¯__call__
+°&call_and_return_all_conditional_losses
'°"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°

flayers
glayer_metrics
hnon_trainable_variables
$regularization_losses
%	variables
&trainable_variables
imetrics
jlayer_regularization_losses
±__call__
+²&call_and_return_all_conditional_losses
'²"call_and_return_conditional_losses"
_generic_user_object
$:"
ÀT2dense_201/kernel
:2dense_201/bias
 "
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
°

klayers
llayer_metrics
mnon_trainable_variables
*regularization_losses
+	variables
,trainable_variables
nmetrics
olayer_regularization_losses
³__call__
+´&call_and_return_all_conditional_losses
'´"call_and_return_conditional_losses"
_generic_user_object
#:!	@2dense_202/kernel
:@2dense_202/bias
 "
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
°

players
qlayer_metrics
rnon_trainable_variables
0regularization_losses
1	variables
2trainable_variables
smetrics
tlayer_regularization_losses
µ__call__
+¶&call_and_return_all_conditional_losses
'¶"call_and_return_conditional_losses"
_generic_user_object
": @
2dense_203/kernel
:
2dense_203/bias
 "
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
°

ulayers
vlayer_metrics
wnon_trainable_variables
6regularization_losses
7	variables
8trainable_variables
xmetrics
ylayer_regularization_losses
·__call__
+¸&call_and_return_all_conditional_losses
'¸"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
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
trackable_dict_wrapper
 "
trackable_list_wrapper
.
z0
{1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
|
_state_var"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²

}layers
~layer_metrics
non_trainable_variables
Eregularization_losses
F	variables
Gtrainable_variables
metrics
 layer_regularization_losses
º__call__
+»&call_and_return_all_conditional_losses
'»"call_and_return_conditional_losses"
_generic_user_object
/

_state_var"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
layers
layer_metrics
non_trainable_variables
Jregularization_losses
K	variables
Ltrainable_variables
metrics
 layer_regularization_losses
¼__call__
+½&call_and_return_all_conditional_losses
'½"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
layers
layer_metrics
non_trainable_variables
Nregularization_losses
O	variables
Ptrainable_variables
metrics
 layer_regularization_losses
¾__call__
+¿&call_and_return_all_conditional_losses
'¿"call_and_return_conditional_losses"
_generic_user_object
5
0
1
2"
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
¿

total

count
	variables
	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}


total

count

_fn_kwargs
	variables
	keras_api"¿
_tf_keras_metric¤{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}}
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
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
0:.@2Adam/conv2d_101/kernel/m
": @2Adam/conv2d_101/bias/m
):'
ÀT2Adam/dense_201/kernel/m
": 2Adam/dense_201/bias/m
(:&	@2Adam/dense_202/kernel/m
!:@2Adam/dense_202/bias/m
':%@
2Adam/dense_203/kernel/m
!:
2Adam/dense_203/bias/m
0:.@2Adam/conv2d_101/kernel/v
": @2Adam/conv2d_101/bias/v
):'
ÀT2Adam/dense_201/kernel/v
": 2Adam/dense_201/bias/v
(:&	@2Adam/dense_202/kernel/v
!:@2Adam/dense_202/bias/v
':%@
2Adam/dense_203/kernel/v
!:
2Adam/dense_203/bias/v
º2·
;__inference_SimpleModel_LessDense_avg_layer_call_fn_1737130
;__inference_SimpleModel_LessDense_avg_layer_call_fn_1737468
;__inference_SimpleModel_LessDense_avg_layer_call_fn_1737447
;__inference_SimpleModel_LessDense_avg_layer_call_fn_1737179À
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
kwonlydefaultsª 
annotationsª *
 
2þ
"__inference__wrapped_model_1736498×
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
annotationsª *G¢D
B?
 Training_Data_Augmentation_inputÿÿÿÿÿÿÿÿÿ
¦2£
V__inference_SimpleModel_LessDense_avg_layer_call_and_return_conditional_losses_1737388
V__inference_SimpleModel_LessDense_avg_layer_call_and_return_conditional_losses_1737424
V__inference_SimpleModel_LessDense_avg_layer_call_and_return_conditional_losses_1737048
V__inference_SimpleModel_LessDense_avg_layer_call_and_return_conditional_losses_1737076À
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
kwonlydefaultsª 
annotationsª *
 
¾2»
<__inference_Training_Data_Augmentation_layer_call_fn_1737623
<__inference_Training_Data_Augmentation_layer_call_fn_1737618
<__inference_Training_Data_Augmentation_layer_call_fn_1736863
<__inference_Training_Data_Augmentation_layer_call_fn_1736851À
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
kwonlydefaultsª 
annotationsª *
 
ª2§
W__inference_Training_Data_Augmentation_layer_call_and_return_conditional_losses_1736827
W__inference_Training_Data_Augmentation_layer_call_and_return_conditional_losses_1736834
W__inference_Training_Data_Augmentation_layer_call_and_return_conditional_losses_1737611
W__inference_Training_Data_Augmentation_layer_call_and_return_conditional_losses_1737607À
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
kwonlydefaultsª 
annotationsª *
 
Ö2Ó
,__inference_conv2d_101_layer_call_fn_1737643¢
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
annotationsª *
 
ñ2î
G__inference_conv2d_101_layer_call_and_return_conditional_losses_1737634¢
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
annotationsª *
 
2
6__inference_average_pooling2d_46_layer_call_fn_1736875à
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
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
¹2¶
Q__inference_average_pooling2d_46_layer_call_and_return_conditional_losses_1736869à
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
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
-__inference_dropout_101_layer_call_fn_1737670
-__inference_dropout_101_layer_call_fn_1737665´
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
kwonlydefaultsª 
annotationsª *
 
Î2Ë
H__inference_dropout_101_layer_call_and_return_conditional_losses_1737660
H__inference_dropout_101_layer_call_and_return_conditional_losses_1737655´
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
kwonlydefaultsª 
annotationsª *
 
Ö2Ó
,__inference_flatten_67_layer_call_fn_1737681¢
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
annotationsª *
 
ñ2î
G__inference_flatten_67_layer_call_and_return_conditional_losses_1737676¢
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
annotationsª *
 
Õ2Ò
+__inference_dense_201_layer_call_fn_1737701¢
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
annotationsª *
 
ð2í
F__inference_dense_201_layer_call_and_return_conditional_losses_1737692¢
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
annotationsª *
 
Õ2Ò
+__inference_dense_202_layer_call_fn_1737721¢
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
annotationsª *
 
ð2í
F__inference_dense_202_layer_call_and_return_conditional_losses_1737712¢
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
annotationsª *
 
Õ2Ò
+__inference_dense_203_layer_call_fn_1737741¢
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
annotationsª *
 
ð2í
F__inference_dense_203_layer_call_and_return_conditional_losses_1737732¢
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
annotationsª *
 
MBK
%__inference_signature_wrapper_1737210 Training_Data_Augmentation_input
ö2ó
-__inference_random_flip_layer_call_fn_1737778
-__inference_random_flip_layer_call_fn_1737849
-__inference_random_flip_layer_call_fn_1737783
-__inference_random_flip_layer_call_fn_1737844´
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
kwonlydefaultsª 
annotationsª *
 
â2ß
H__inference_random_flip_layer_call_and_return_conditional_losses_1737835
H__inference_random_flip_layer_call_and_return_conditional_losses_1737773
H__inference_random_flip_layer_call_and_return_conditional_losses_1737769
H__inference_random_flip_layer_call_and_return_conditional_losses_1737839´
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
kwonlydefaultsª 
annotationsª *
 
 2
1__inference_random_rotation_layer_call_fn_1737966
1__inference_random_rotation_layer_call_fn_1737971´
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
kwonlydefaultsª 
annotationsª *
 
Ö2Ó
L__inference_random_rotation_layer_call_and_return_conditional_losses_1737955
L__inference_random_rotation_layer_call_and_return_conditional_losses_1737959´
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
kwonlydefaultsª 
annotationsª *
 
 2
1__inference_random_contrast_layer_call_fn_1737993
1__inference_random_contrast_layer_call_fn_1737998´
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
kwonlydefaultsª 
annotationsª *
 
Ö2Ó
L__inference_random_contrast_layer_call_and_return_conditional_losses_1737984
L__inference_random_contrast_layer_call_and_return_conditional_losses_1737988´
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
kwonlydefaultsª 
annotationsª *
 é
V__inference_SimpleModel_LessDense_avg_layer_call_and_return_conditional_losses_1737048
()./45Y¢V
O¢L
B?
 Training_Data_Augmentation_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 ç
V__inference_SimpleModel_LessDense_avg_layer_call_and_return_conditional_losses_1737076()./45Y¢V
O¢L
B?
 Training_Data_Augmentation_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 Î
V__inference_SimpleModel_LessDense_avg_layer_call_and_return_conditional_losses_1737388t
()./45?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 Ì
V__inference_SimpleModel_LessDense_avg_layer_call_and_return_conditional_losses_1737424r()./45?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 Á
;__inference_SimpleModel_LessDense_avg_layer_call_fn_1737130
()./45Y¢V
O¢L
B?
 Training_Data_Augmentation_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
¾
;__inference_SimpleModel_LessDense_avg_layer_call_fn_1737179()./45Y¢V
O¢L
B?
 Training_Data_Augmentation_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
¦
;__inference_SimpleModel_LessDense_avg_layer_call_fn_1737447g
()./45?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
¤
;__inference_SimpleModel_LessDense_avg_layer_call_fn_1737468e()./45?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
Ú
W__inference_Training_Data_Augmentation_layer_call_and_return_conditional_losses_1736827J¢G
@¢=
30
random_flip_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 Ö
W__inference_Training_Data_Augmentation_layer_call_and_return_conditional_losses_1736834{J¢G
@¢=
30
random_flip_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 Ï
W__inference_Training_Data_Augmentation_layer_call_and_return_conditional_losses_1737607t?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 Ë
W__inference_Training_Data_Augmentation_layer_call_and_return_conditional_losses_1737611p?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 ²
<__inference_Training_Data_Augmentation_layer_call_fn_1736851rJ¢G
@¢=
30
random_flip_inputÿÿÿÿÿÿÿÿÿ
p

 
ª " ÿÿÿÿÿÿÿÿÿ®
<__inference_Training_Data_Augmentation_layer_call_fn_1736863nJ¢G
@¢=
30
random_flip_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª " ÿÿÿÿÿÿÿÿÿ§
<__inference_Training_Data_Augmentation_layer_call_fn_1737618g?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª " ÿÿÿÿÿÿÿÿÿ£
<__inference_Training_Data_Augmentation_layer_call_fn_1737623c?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª " ÿÿÿÿÿÿÿÿÿ»
"__inference__wrapped_model_1736498()./45Q¢N
G¢D
B?
 Training_Data_Augmentation_inputÿÿÿÿÿÿÿÿÿ
ª "5ª2
0
	dense_203# 
	dense_203ÿÿÿÿÿÿÿÿÿ
ô
Q__inference_average_pooling2d_46_layer_call_and_return_conditional_losses_1736869R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ì
6__inference_average_pooling2d_46_layer_call_fn_1736875R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ·
G__inference_conv2d_101_layer_call_and_return_conditional_losses_1737634l7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 
,__inference_conv2d_101_layer_call_fn_1737643_7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª " ÿÿÿÿÿÿÿÿÿ@¨
F__inference_dense_201_layer_call_and_return_conditional_losses_1737692^()0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿÀT
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
+__inference_dense_201_layer_call_fn_1737701Q()0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿÀT
ª "ÿÿÿÿÿÿÿÿÿ§
F__inference_dense_202_layer_call_and_return_conditional_losses_1737712]./0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 
+__inference_dense_202_layer_call_fn_1737721P./0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ@¦
F__inference_dense_203_layer_call_and_return_conditional_losses_1737732\45/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 ~
+__inference_dense_203_layer_call_fn_1737741O45/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ
¸
H__inference_dropout_101_layer_call_and_return_conditional_losses_1737655l;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 ¸
H__inference_dropout_101_layer_call_and_return_conditional_losses_1737660l;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 
-__inference_dropout_101_layer_call_fn_1737665_;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p
ª " ÿÿÿÿÿÿÿÿÿ@
-__inference_dropout_101_layer_call_fn_1737670_;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª " ÿÿÿÿÿÿÿÿÿ@¬
G__inference_flatten_67_layer_call_and_return_conditional_losses_1737676a7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "&¢#

0ÿÿÿÿÿÿÿÿÿÀT
 
,__inference_flatten_67_layer_call_fn_1737681T7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿÀT¼
L__inference_random_contrast_layer_call_and_return_conditional_losses_1737984l;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 ¼
L__inference_random_contrast_layer_call_and_return_conditional_losses_1737988l;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 
1__inference_random_contrast_layer_call_fn_1737993_;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ
p
ª " ÿÿÿÿÿÿÿÿÿ
1__inference_random_contrast_layer_call_fn_1737998_;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ
p 
ª " ÿÿÿÿÿÿÿÿÿ¸
H__inference_random_flip_layer_call_and_return_conditional_losses_1737769l;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 ¸
H__inference_random_flip_layer_call_and_return_conditional_losses_1737773l;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 ï
H__inference_random_flip_layer_call_and_return_conditional_losses_1737835¢V¢S
L¢I
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ï
H__inference_random_flip_layer_call_and_return_conditional_losses_1737839¢V¢S
L¢I
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
-__inference_random_flip_layer_call_fn_1737778_;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ
p
ª " ÿÿÿÿÿÿÿÿÿ
-__inference_random_flip_layer_call_fn_1737783_;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ
p 
ª " ÿÿÿÿÿÿÿÿÿÇ
-__inference_random_flip_layer_call_fn_1737844V¢S
L¢I
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÇ
-__inference_random_flip_layer_call_fn_1737849V¢S
L¢I
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ
L__inference_random_rotation_layer_call_and_return_conditional_losses_1737955p;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 ¼
L__inference_random_rotation_layer_call_and_return_conditional_losses_1737959l;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 
1__inference_random_rotation_layer_call_fn_1737966c;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ
p
ª " ÿÿÿÿÿÿÿÿÿ
1__inference_random_rotation_layer_call_fn_1737971_;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ
p 
ª " ÿÿÿÿÿÿÿÿÿâ
%__inference_signature_wrapper_1737210¸()./45u¢r
¢ 
kªh
f
 Training_Data_Augmentation_inputB?
 Training_Data_Augmentation_inputÿÿÿÿÿÿÿÿÿ"5ª2
0
	dense_203# 
	dense_203ÿÿÿÿÿÿÿÿÿ
