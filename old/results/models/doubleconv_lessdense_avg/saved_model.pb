Ê
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
 "serve*2.3.02unknown8°Ü

conv2d_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_nameconv2d_7/kernel
{
#conv2d_7/kernel/Read/ReadVariableOpReadVariableOpconv2d_7/kernel*&
_output_shapes
:@*
dtype0
r
conv2d_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_7/bias
k
!conv2d_7/bias/Read/ReadVariableOpReadVariableOpconv2d_7/bias*
_output_shapes
:@*
dtype0

conv2d_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@* 
shared_nameconv2d_8/kernel
{
#conv2d_8/kernel/Read/ReadVariableOpReadVariableOpconv2d_8/kernel*&
_output_shapes
:@@*
dtype0
r
conv2d_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_8/bias
k
!conv2d_8/bias/Read/ReadVariableOpReadVariableOpconv2d_8/bias*
_output_shapes
:@*
dtype0
|
dense_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
* 
shared_namedense_15/kernel
u
#dense_15/kernel/Read/ReadVariableOpReadVariableOpdense_15/kernel* 
_output_shapes
:
*
dtype0
s
dense_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_15/bias
l
!dense_15/bias/Read/ReadVariableOpReadVariableOpdense_15/bias*
_output_shapes	
:*
dtype0
{
dense_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@* 
shared_namedense_16/kernel
t
#dense_16/kernel/Read/ReadVariableOpReadVariableOpdense_16/kernel*
_output_shapes
:	@*
dtype0
r
dense_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_16/bias
k
!dense_16/bias/Read/ReadVariableOpReadVariableOpdense_16/bias*
_output_shapes
:@*
dtype0
z
dense_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@
* 
shared_namedense_17/kernel
s
#dense_17/kernel/Read/ReadVariableOpReadVariableOpdense_17/kernel*
_output_shapes

:@
*
dtype0
r
dense_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_17/bias
k
!dense_17/bias/Read/ReadVariableOpReadVariableOpdense_17/bias*
_output_shapes
:
*
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

NoOpNoOp
í0
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*¨0
value0B0 B0
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
 
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

Klayers
Llayer_metrics
Mnon_trainable_variables
regularization_losses
	variables
trainable_variables
Nmetrics
Olayer_regularization_losses
 
\
P_rng
Qregularization_losses
R	variables
Strainable_variables
T	keras_api
\
U_rng
Vregularization_losses
W	variables
Xtrainable_variables
Y	keras_api
R
Zregularization_losses
[	variables
\trainable_variables
]	keras_api
 
 
 
­

^layers
_layer_metrics
`non_trainable_variables
regularization_losses
	variables
trainable_variables
ametrics
blayer_regularization_losses
[Y
VARIABLE_VALUEconv2d_7/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_7/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
­

clayers
dlayer_metrics
enon_trainable_variables
regularization_losses
	variables
trainable_variables
fmetrics
glayer_regularization_losses
 
 
 
­

hlayers
ilayer_metrics
jnon_trainable_variables
regularization_losses
 	variables
!trainable_variables
kmetrics
llayer_regularization_losses
 
 
 
­

mlayers
nlayer_metrics
onon_trainable_variables
#regularization_losses
$	variables
%trainable_variables
pmetrics
qlayer_regularization_losses
[Y
VARIABLE_VALUEconv2d_8/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_8/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

'0
(1

'0
(1
­

rlayers
slayer_metrics
tnon_trainable_variables
)regularization_losses
*	variables
+trainable_variables
umetrics
vlayer_regularization_losses
 
 
 
­

wlayers
xlayer_metrics
ynon_trainable_variables
-regularization_losses
.	variables
/trainable_variables
zmetrics
{layer_regularization_losses
 
 
 
®

|layers
}layer_metrics
~non_trainable_variables
1regularization_losses
2	variables
3trainable_variables
metrics
 layer_regularization_losses
 
 
 
²
layers
layer_metrics
non_trainable_variables
5regularization_losses
6	variables
7trainable_variables
metrics
 layer_regularization_losses
[Y
VARIABLE_VALUEdense_15/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_15/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

90
:1

90
:1
²
layers
layer_metrics
non_trainable_variables
;regularization_losses
<	variables
=trainable_variables
metrics
 layer_regularization_losses
[Y
VARIABLE_VALUEdense_16/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_16/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
@1

?0
@1
²
layers
layer_metrics
non_trainable_variables
Aregularization_losses
B	variables
Ctrainable_variables
metrics
 layer_regularization_losses
[Y
VARIABLE_VALUEdense_17/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_17/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

E0
F1

E0
F1
²
layers
layer_metrics
non_trainable_variables
Gregularization_losses
H	variables
Itrainable_variables
metrics
 layer_regularization_losses
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
 
 


_state_var
 
 
 
²
layers
layer_metrics
non_trainable_variables
Qregularization_losses
R	variables
Strainable_variables
metrics
 layer_regularization_losses


_state_var
 
 
 
²
layers
layer_metrics
non_trainable_variables
Vregularization_losses
W	variables
Xtrainable_variables
metrics
  layer_regularization_losses
 
 
 
²
¡layers
¢layer_metrics
£non_trainable_variables
Zregularization_losses
[	variables
\trainable_variables
¤metrics
 ¥layer_regularization_losses
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
£
0serving_default_Training_Data_Augmentation_inputPlaceholder*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*$
shape:ÿÿÿÿÿÿÿÿÿ
þ
StatefulPartitionedCallStatefulPartitionedCall0serving_default_Training_Data_Augmentation_inputconv2d_7/kernelconv2d_7/biasconv2d_8/kernelconv2d_8/biasdense_15/kerneldense_15/biasdense_16/kerneldense_16/biasdense_17/kerneldense_17/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference_signature_wrapper_11693
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ì
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#conv2d_7/kernel/Read/ReadVariableOp!conv2d_7/bias/Read/ReadVariableOp#conv2d_8/kernel/Read/ReadVariableOp!conv2d_8/bias/Read/ReadVariableOp#dense_15/kernel/Read/ReadVariableOp!dense_15/bias/Read/ReadVariableOp#dense_16/kernel/Read/ReadVariableOp!dense_16/bias/Read/ReadVariableOp#dense_17/kernel/Read/ReadVariableOp!dense_17/bias/Read/ReadVariableOpVariable/Read/ReadVariableOpVariable_1/Read/ReadVariableOpConst*
Tin
2		*
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
GPU 2J 8 *'
f"R 
__inference__traced_save_12630
×
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_7/kernelconv2d_7/biasconv2d_8/kernelconv2d_8/biasdense_15/kerneldense_15/biasdense_16/kerneldense_16/biasdense_17/kerneldense_17/biasVariable
Variable_1*
Tin
2*
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
GPU 2J 8 **
f%R#
!__inference__traced_restore_12676 
Â
c
D__inference_dropout_7_layer_call_and_return_conditional_losses_11334

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
Þ
}
(__inference_dense_15_layer_call_fn_12264

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallô
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
GPU 2J 8 *L
fGRE
C__inference_dense_15_layer_call_and_return_conditional_losses_114352
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
(
e
F__inference_random_flip_layer_call_and_return_conditional_losses_12398

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
°
«
C__inference_dense_17_layer_call_and_return_conditional_losses_12295

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
ç
b
D__inference_dropout_8_layer_call_and_return_conditional_losses_12223

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
	
«
C__inference_conv2d_7_layer_call_and_return_conditional_losses_11305

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
ú
}
(__inference_conv2d_8_layer_call_fn_12206

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallû
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_8_layer_call_and_return_conditional_losses_113632
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
	

#__inference_signature_wrapper_11693$
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
identity¢StatefulPartitionedCallÒ
StatefulPartitionedCallStatefulPartitionedCall training_data_augmentation_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__wrapped_model_108862
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿ::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:q m
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:
_user_specified_name" Training_Data_Augmentation_input
 E

 __inference__wrapped_model_10886$
 training_data_augmentation_inputD
@doubleconv_lessdense_avg_conv2d_7_conv2d_readvariableop_resourceE
Adoubleconv_lessdense_avg_conv2d_7_biasadd_readvariableop_resourceD
@doubleconv_lessdense_avg_conv2d_8_conv2d_readvariableop_resourceE
Adoubleconv_lessdense_avg_conv2d_8_biasadd_readvariableop_resourceD
@doubleconv_lessdense_avg_dense_15_matmul_readvariableop_resourceE
Adoubleconv_lessdense_avg_dense_15_biasadd_readvariableop_resourceD
@doubleconv_lessdense_avg_dense_16_matmul_readvariableop_resourceE
Adoubleconv_lessdense_avg_dense_16_biasadd_readvariableop_resourceD
@doubleconv_lessdense_avg_dense_17_matmul_readvariableop_resourceE
Adoubleconv_lessdense_avg_dense_17_biasadd_readvariableop_resource
identityû
7DoubleConv_LessDense_avg/conv2d_7/Conv2D/ReadVariableOpReadVariableOp@doubleconv_lessdense_avg_conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype029
7DoubleConv_LessDense_avg/conv2d_7/Conv2D/ReadVariableOp¤
(DoubleConv_LessDense_avg/conv2d_7/Conv2DConv2D training_data_augmentation_input?DoubleConv_LessDense_avg/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2*
(DoubleConv_LessDense_avg/conv2d_7/Conv2Dò
8DoubleConv_LessDense_avg/conv2d_7/BiasAdd/ReadVariableOpReadVariableOpAdoubleconv_lessdense_avg_conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02:
8DoubleConv_LessDense_avg/conv2d_7/BiasAdd/ReadVariableOp
)DoubleConv_LessDense_avg/conv2d_7/BiasAddBiasAdd1DoubleConv_LessDense_avg/conv2d_7/Conv2D:output:0@DoubleConv_LessDense_avg/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2+
)DoubleConv_LessDense_avg/conv2d_7/BiasAddÆ
&DoubleConv_LessDense_avg/conv2d_7/ReluRelu2DoubleConv_LessDense_avg/conv2d_7/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2(
&DoubleConv_LessDense_avg/conv2d_7/Relu£
4DoubleConv_LessDense_avg/average_pooling2d_2/AvgPoolAvgPool4DoubleConv_LessDense_avg/conv2d_7/Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
26
4DoubleConv_LessDense_avg/average_pooling2d_2/AvgPoolß
+DoubleConv_LessDense_avg/dropout_7/IdentityIdentity=DoubleConv_LessDense_avg/average_pooling2d_2/AvgPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2-
+DoubleConv_LessDense_avg/dropout_7/Identityû
7DoubleConv_LessDense_avg/conv2d_8/Conv2D/ReadVariableOpReadVariableOp@doubleconv_lessdense_avg_conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype029
7DoubleConv_LessDense_avg/conv2d_8/Conv2D/ReadVariableOp¸
(DoubleConv_LessDense_avg/conv2d_8/Conv2DConv2D4DoubleConv_LessDense_avg/dropout_7/Identity:output:0?DoubleConv_LessDense_avg/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2*
(DoubleConv_LessDense_avg/conv2d_8/Conv2Dò
8DoubleConv_LessDense_avg/conv2d_8/BiasAdd/ReadVariableOpReadVariableOpAdoubleconv_lessdense_avg_conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02:
8DoubleConv_LessDense_avg/conv2d_8/BiasAdd/ReadVariableOp
)DoubleConv_LessDense_avg/conv2d_8/BiasAddBiasAdd1DoubleConv_LessDense_avg/conv2d_8/Conv2D:output:0@DoubleConv_LessDense_avg/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2+
)DoubleConv_LessDense_avg/conv2d_8/BiasAddÆ
&DoubleConv_LessDense_avg/conv2d_8/ReluRelu2DoubleConv_LessDense_avg/conv2d_8/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2(
&DoubleConv_LessDense_avg/conv2d_8/Relu£
4DoubleConv_LessDense_avg/average_pooling2d_3/AvgPoolAvgPool4DoubleConv_LessDense_avg/conv2d_8/Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
26
4DoubleConv_LessDense_avg/average_pooling2d_3/AvgPoolß
+DoubleConv_LessDense_avg/dropout_8/IdentityIdentity=DoubleConv_LessDense_avg/average_pooling2d_3/AvgPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2-
+DoubleConv_LessDense_avg/dropout_8/Identity¥
(DoubleConv_LessDense_avg/flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ 	  2*
(DoubleConv_LessDense_avg/flatten_5/Constÿ
*DoubleConv_LessDense_avg/flatten_5/ReshapeReshape4DoubleConv_LessDense_avg/dropout_8/Identity:output:01DoubleConv_LessDense_avg/flatten_5/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*DoubleConv_LessDense_avg/flatten_5/Reshapeõ
7DoubleConv_LessDense_avg/dense_15/MatMul/ReadVariableOpReadVariableOp@doubleconv_lessdense_avg_dense_15_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype029
7DoubleConv_LessDense_avg/dense_15/MatMul/ReadVariableOp
(DoubleConv_LessDense_avg/dense_15/MatMulMatMul3DoubleConv_LessDense_avg/flatten_5/Reshape:output:0?DoubleConv_LessDense_avg/dense_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(DoubleConv_LessDense_avg/dense_15/MatMuló
8DoubleConv_LessDense_avg/dense_15/BiasAdd/ReadVariableOpReadVariableOpAdoubleconv_lessdense_avg_dense_15_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02:
8DoubleConv_LessDense_avg/dense_15/BiasAdd/ReadVariableOp
)DoubleConv_LessDense_avg/dense_15/BiasAddBiasAdd2DoubleConv_LessDense_avg/dense_15/MatMul:product:0@DoubleConv_LessDense_avg/dense_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)DoubleConv_LessDense_avg/dense_15/BiasAdd¿
&DoubleConv_LessDense_avg/dense_15/ReluRelu2DoubleConv_LessDense_avg/dense_15/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&DoubleConv_LessDense_avg/dense_15/Reluô
7DoubleConv_LessDense_avg/dense_16/MatMul/ReadVariableOpReadVariableOp@doubleconv_lessdense_avg_dense_16_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype029
7DoubleConv_LessDense_avg/dense_16/MatMul/ReadVariableOp
(DoubleConv_LessDense_avg/dense_16/MatMulMatMul4DoubleConv_LessDense_avg/dense_15/Relu:activations:0?DoubleConv_LessDense_avg/dense_16/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2*
(DoubleConv_LessDense_avg/dense_16/MatMulò
8DoubleConv_LessDense_avg/dense_16/BiasAdd/ReadVariableOpReadVariableOpAdoubleconv_lessdense_avg_dense_16_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02:
8DoubleConv_LessDense_avg/dense_16/BiasAdd/ReadVariableOp
)DoubleConv_LessDense_avg/dense_16/BiasAddBiasAdd2DoubleConv_LessDense_avg/dense_16/MatMul:product:0@DoubleConv_LessDense_avg/dense_16/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2+
)DoubleConv_LessDense_avg/dense_16/BiasAdd¾
&DoubleConv_LessDense_avg/dense_16/ReluRelu2DoubleConv_LessDense_avg/dense_16/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2(
&DoubleConv_LessDense_avg/dense_16/Reluó
7DoubleConv_LessDense_avg/dense_17/MatMul/ReadVariableOpReadVariableOp@doubleconv_lessdense_avg_dense_17_matmul_readvariableop_resource*
_output_shapes

:@
*
dtype029
7DoubleConv_LessDense_avg/dense_17/MatMul/ReadVariableOp
(DoubleConv_LessDense_avg/dense_17/MatMulMatMul4DoubleConv_LessDense_avg/dense_16/Relu:activations:0?DoubleConv_LessDense_avg/dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2*
(DoubleConv_LessDense_avg/dense_17/MatMulò
8DoubleConv_LessDense_avg/dense_17/BiasAdd/ReadVariableOpReadVariableOpAdoubleconv_lessdense_avg_dense_17_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02:
8DoubleConv_LessDense_avg/dense_17/BiasAdd/ReadVariableOp
)DoubleConv_LessDense_avg/dense_17/BiasAddBiasAdd2DoubleConv_LessDense_avg/dense_17/MatMul:product:0@DoubleConv_LessDense_avg/dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2+
)DoubleConv_LessDense_avg/dense_17/BiasAddÇ
)DoubleConv_LessDense_avg/dense_17/SoftmaxSoftmax2DoubleConv_LessDense_avg/dense_17/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2+
)DoubleConv_LessDense_avg/dense_17/Softmax
IdentityIdentity3DoubleConv_LessDense_avg/dense_17/Softmax:softmax:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿ:::::::::::q m
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:
_user_specified_name" Training_Data_Augmentation_input
ø	
®
8__inference_DoubleConv_LessDense_avg_layer_call_fn_11606$
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
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCall training_data_augmentation_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_DoubleConv_LessDense_avg_layer_call_and_return_conditional_losses_115812
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿ:::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:q m
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:
_user_specified_name" Training_Data_Augmentation_input
ç
b
D__inference_dropout_8_layer_call_and_return_conditional_losses_11397

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ù

U__inference_Training_Data_Augmentation_layer_call_and_return_conditional_losses_11215
random_flip_input
random_rotation_11181
identity¢'random_contrast/StatefulPartitionedCall¢#random_flip/StatefulPartitionedCall¢'random_rotation/StatefulPartitionedCall
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
GPU 2J 8 *O
fJRH
F__inference_random_flip_layer_call_and_return_conditional_losses_110412%
#random_flip/StatefulPartitionedCallÆ
'random_rotation/StatefulPartitionedCallStatefulPartitionedCall,random_flip/StatefulPartitionedCall:output:0random_rotation_11181*
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
GPU 2J 8 *S
fNRL
J__inference_random_rotation_layer_call_and_return_conditional_losses_111632)
'random_rotation/StatefulPartitionedCallØ
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
GPU 2J 8 *S
fNRL
J__inference_random_contrast_layer_call_and_return_conditional_losses_111972)
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
5

!__inference__traced_restore_12676
file_prefix$
 assignvariableop_conv2d_7_kernel$
 assignvariableop_1_conv2d_7_bias&
"assignvariableop_2_conv2d_8_kernel$
 assignvariableop_3_conv2d_8_bias&
"assignvariableop_4_dense_15_kernel$
 assignvariableop_5_dense_15_bias&
"assignvariableop_6_dense_16_kernel$
 assignvariableop_7_dense_16_bias&
"assignvariableop_8_dense_17_kernel$
 assignvariableop_9_dense_17_bias 
assignvariableop_10_variable"
assignvariableop_11_variable_1
identity_13¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_2¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9Å
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ñ
valueÇBÄB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB:layer-0/layer-0/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUEB:layer-0/layer-1/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names¨
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*-
value$B"B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesì
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*H
_output_shapes6
4:::::::::::::*
dtypes
2		2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOp assignvariableop_conv2d_7_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¥
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv2d_7_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2§
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_8_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¥
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_8_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4§
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_15_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5¥
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_15_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6§
AssignVariableOp_6AssignVariableOp"assignvariableop_6_dense_16_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7¥
AssignVariableOp_7AssignVariableOp assignvariableop_7_dense_16_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8§
AssignVariableOp_8AssignVariableOp"assignvariableop_8_dense_17_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9¥
AssignVariableOp_9AssignVariableOp assignvariableop_9_dense_17_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_10¤
AssignVariableOp_10AssignVariableOpassignvariableop_10_variableIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_11¦
AssignVariableOp_11AssignVariableOpassignvariableop_11_variable_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_119
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpæ
Identity_12Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_12Ù
Identity_13IdentityIdentity_12:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_13"#
identity_13Identity_13:output:0*E
_input_shapes4
2: ::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
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
²
E
)__inference_dropout_7_layer_call_fn_12186

inputs
identityÊ
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
GPU 2J 8 *M
fHRF
D__inference_dropout_7_layer_call_and_return_conditional_losses_113392
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
å
á
S__inference_DoubleConv_LessDense_avg_layer_call_and_return_conditional_losses_11887

inputsX
Ttraining_data_augmentation_random_rotation_stateful_uniform_statefuluniform_resource+
'conv2d_7_conv2d_readvariableop_resource,
(conv2d_7_biasadd_readvariableop_resource+
'conv2d_8_conv2d_readvariableop_resource,
(conv2d_8_biasadd_readvariableop_resource+
'dense_15_matmul_readvariableop_resource,
(dense_15_biasadd_readvariableop_resource+
'dense_16_matmul_readvariableop_resource,
(dense_16_biasadd_readvariableop_resource+
'dense_17_matmul_readvariableop_resource,
(dense_17_biasadd_readvariableop_resource
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
CTraining_Data_Augmentation/random_contrast/adjust_contrast/Identity°
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02 
conv2d_7/Conv2D/ReadVariableOp
conv2d_7/Conv2DConv2DLTraining_Data_Augmentation/random_contrast/adjust_contrast/Identity:output:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
conv2d_7/Conv2D§
conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_7/BiasAdd/ReadVariableOp¬
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d_7/BiasAdd{
conv2d_7/ReluReluconv2d_7/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d_7/ReluØ
average_pooling2d_2/AvgPoolAvgPoolconv2d_7/Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
2
average_pooling2d_2/AvgPoolw
dropout_7/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?2
dropout_7/dropout/Const·
dropout_7/dropout/MulMul$average_pooling2d_2/AvgPool:output:0 dropout_7/dropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_7/dropout/Mul
dropout_7/dropout/ShapeShape$average_pooling2d_2/AvgPool:output:0*
T0*
_output_shapes
:2
dropout_7/dropout/ShapeÚ
.dropout_7/dropout/random_uniform/RandomUniformRandomUniform dropout_7/dropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype020
.dropout_7/dropout/random_uniform/RandomUniform
 dropout_7/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2"
 dropout_7/dropout/GreaterEqual/yî
dropout_7/dropout/GreaterEqualGreaterEqual7dropout_7/dropout/random_uniform/RandomUniform:output:0)dropout_7/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2 
dropout_7/dropout/GreaterEqual¥
dropout_7/dropout/CastCast"dropout_7/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_7/dropout/Castª
dropout_7/dropout/Mul_1Muldropout_7/dropout/Mul:z:0dropout_7/dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_7/dropout/Mul_1°
conv2d_8/Conv2D/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_8/Conv2D/ReadVariableOpÔ
conv2d_8/Conv2DConv2Ddropout_7/dropout/Mul_1:z:0&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
conv2d_8/Conv2D§
conv2d_8/BiasAdd/ReadVariableOpReadVariableOp(conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_8/BiasAdd/ReadVariableOp¬
conv2d_8/BiasAddBiasAddconv2d_8/Conv2D:output:0'conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d_8/BiasAdd{
conv2d_8/ReluReluconv2d_8/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d_8/ReluØ
average_pooling2d_3/AvgPoolAvgPoolconv2d_8/Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
2
average_pooling2d_3/AvgPoolw
dropout_8/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?2
dropout_8/dropout/Const·
dropout_8/dropout/MulMul$average_pooling2d_3/AvgPool:output:0 dropout_8/dropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_8/dropout/Mul
dropout_8/dropout/ShapeShape$average_pooling2d_3/AvgPool:output:0*
T0*
_output_shapes
:2
dropout_8/dropout/ShapeÚ
.dropout_8/dropout/random_uniform/RandomUniformRandomUniform dropout_8/dropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype020
.dropout_8/dropout/random_uniform/RandomUniform
 dropout_8/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>2"
 dropout_8/dropout/GreaterEqual/yî
dropout_8/dropout/GreaterEqualGreaterEqual7dropout_8/dropout/random_uniform/RandomUniform:output:0)dropout_8/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2 
dropout_8/dropout/GreaterEqual¥
dropout_8/dropout/CastCast"dropout_8/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_8/dropout/Castª
dropout_8/dropout/Mul_1Muldropout_8/dropout/Mul:z:0dropout_8/dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_8/dropout/Mul_1s
flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ 	  2
flatten_5/Const
flatten_5/ReshapeReshapedropout_8/dropout/Mul_1:z:0flatten_5/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
flatten_5/Reshapeª
dense_15/MatMul/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02 
dense_15/MatMul/ReadVariableOp£
dense_15/MatMulMatMulflatten_5/Reshape:output:0&dense_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_15/MatMul¨
dense_15/BiasAdd/ReadVariableOpReadVariableOp(dense_15_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_15/BiasAdd/ReadVariableOp¦
dense_15/BiasAddBiasAdddense_15/MatMul:product:0'dense_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_15/BiasAddt
dense_15/ReluReludense_15/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_15/Relu©
dense_16/MatMul/ReadVariableOpReadVariableOp'dense_16_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02 
dense_16/MatMul/ReadVariableOp£
dense_16/MatMulMatMuldense_15/Relu:activations:0&dense_16/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_16/MatMul§
dense_16/BiasAdd/ReadVariableOpReadVariableOp(dense_16_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_16/BiasAdd/ReadVariableOp¥
dense_16/BiasAddBiasAdddense_16/MatMul:product:0'dense_16/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_16/BiasAdds
dense_16/ReluReludense_16/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_16/Relu¨
dense_17/MatMul/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource*
_output_shapes

:@
*
dtype02 
dense_17/MatMul/ReadVariableOp£
dense_17/MatMulMatMuldense_16/Relu:activations:0&dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_17/MatMul§
dense_17/BiasAdd/ReadVariableOpReadVariableOp(dense_17_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_17/BiasAdd/ReadVariableOp¥
dense_17/BiasAddBiasAdddense_17/MatMul:product:0'dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_17/BiasAdd|
dense_17/SoftmaxSoftmaxdense_17/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_17/Softmax¼
IdentityIdentitydense_17/Softmax:softmax:0L^Training_Data_Augmentation/random_rotation/stateful_uniform/StatefulUniform*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿ:::::::::::2
KTraining_Data_Augmentation/random_rotation/stateful_uniform/StatefulUniformKTraining_Data_Augmentation/random_rotation/stateful_uniform/StatefulUniform:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ü
}
(__inference_dense_16_layer_call_fn_12284

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCalló
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
GPU 2J 8 *L
fGRE
C__inference_dense_16_layer_call_and_return_conditional_losses_114622
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
¾
b
)__inference_dropout_8_layer_call_fn_12228

inputs
identity¢StatefulPartitionedCallâ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_8_layer_call_and_return_conditional_losses_113922
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Â
c
D__inference_dropout_8_layer_call_and_return_conditional_losses_11392

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
:ÿÿÿÿÿÿÿÿÿ@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¼
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
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
:ÿÿÿÿÿÿÿÿÿ@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Á
b
F__inference_random_flip_layer_call_and_return_conditional_losses_12360

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
Ù	

8__inference_DoubleConv_LessDense_avg_layer_call_fn_11666$
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
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCall training_data_augmentation_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_DoubleConv_LessDense_avg_layer_call_and_return_conditional_losses_116432
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿ::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:q m
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:
_user_specified_name" Training_Data_Augmentation_input
ô
f
J__inference_random_contrast_layer_call_and_return_conditional_losses_11201

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
Ú
}
(__inference_dense_17_layer_call_fn_12304

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCalló
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
GPU 2J 8 *L
fGRE
C__inference_dense_17_layer_call_and_return_conditional_losses_114892
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
ç
b
D__inference_dropout_7_layer_call_and_return_conditional_losses_11339

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
/

S__inference_DoubleConv_LessDense_avg_layer_call_and_return_conditional_losses_11541$
 training_data_augmentation_input
conv2d_7_11510
conv2d_7_11512
conv2d_8_11517
conv2d_8_11519
dense_15_11525
dense_15_11527
dense_16_11530
dense_16_11532
dense_17_11535
dense_17_11537
identity¢ conv2d_7/StatefulPartitionedCall¢ conv2d_8/StatefulPartitionedCall¢ dense_15/StatefulPartitionedCall¢ dense_16/StatefulPartitionedCall¢ dense_17/StatefulPartitionedCall«
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
GPU 2J 8 *^
fYRW
U__inference_Training_Data_Augmentation_layer_call_and_return_conditional_losses_112482,
*Training_Data_Augmentation/PartitionedCallÆ
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall3Training_Data_Augmentation/PartitionedCall:output:0conv2d_7_11510conv2d_7_11512*
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
GPU 2J 8 *L
fGRE
C__inference_conv2d_7_layer_call_and_return_conditional_losses_113052"
 conv2d_7/StatefulPartitionedCall
#average_pooling2d_2/PartitionedCallPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*
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
GPU 2J 8 *W
fRRP
N__inference_average_pooling2d_2_layer_call_and_return_conditional_losses_112572%
#average_pooling2d_2/PartitionedCall
dropout_7/PartitionedCallPartitionedCall,average_pooling2d_2/PartitionedCall:output:0*
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
GPU 2J 8 *M
fHRF
D__inference_dropout_7_layer_call_and_return_conditional_losses_113392
dropout_7/PartitionedCallµ
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCall"dropout_7/PartitionedCall:output:0conv2d_8_11517conv2d_8_11519*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_8_layer_call_and_return_conditional_losses_113632"
 conv2d_8/StatefulPartitionedCall
#average_pooling2d_3/PartitionedCallPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_average_pooling2d_3_layer_call_and_return_conditional_losses_112692%
#average_pooling2d_3/PartitionedCall
dropout_8/PartitionedCallPartitionedCall,average_pooling2d_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_8_layer_call_and_return_conditional_losses_113972
dropout_8/PartitionedCalló
flatten_5/PartitionedCallPartitionedCall"dropout_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_flatten_5_layer_call_and_return_conditional_losses_114162
flatten_5/PartitionedCall®
 dense_15/StatefulPartitionedCallStatefulPartitionedCall"flatten_5/PartitionedCall:output:0dense_15_11525dense_15_11527*
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
GPU 2J 8 *L
fGRE
C__inference_dense_15_layer_call_and_return_conditional_losses_114352"
 dense_15/StatefulPartitionedCall´
 dense_16/StatefulPartitionedCallStatefulPartitionedCall)dense_15/StatefulPartitionedCall:output:0dense_16_11530dense_16_11532*
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
GPU 2J 8 *L
fGRE
C__inference_dense_16_layer_call_and_return_conditional_losses_114622"
 dense_16/StatefulPartitionedCall´
 dense_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0dense_17_11535dense_17_11537*
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
GPU 2J 8 *L
fGRE
C__inference_dense_17_layer_call_and_return_conditional_losses_114892"
 dense_17/StatefulPartitionedCall¬
IdentityIdentity)dense_17/StatefulPartitionedCall:output:0!^conv2d_7/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿ::::::::::2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall:q m
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:
_user_specified_name" Training_Data_Augmentation_input
¼
`
D__inference_flatten_5_layer_call_and_return_conditional_losses_12239

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ 	  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¹
i
J__inference_random_contrast_layer_call_and_return_conditional_losses_12547

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
	
«
C__inference_conv2d_8_layer_call_and_return_conditional_losses_12197

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
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
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@:::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

j
N__inference_average_pooling2d_3_layer_call_and_return_conditional_losses_11269

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
¾
K
/__inference_random_contrast_layer_call_fn_12561

inputs
identityÐ
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
GPU 2J 8 *S
fNRL
J__inference_random_contrast_layer_call_and_return_conditional_losses_112012
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
ª	

8__inference_DoubleConv_LessDense_avg_layer_call_fn_11959

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
identity¢StatefulPartitionedCall÷
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_DoubleConv_LessDense_avg_layer_call_and_return_conditional_losses_115812
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿ:::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ô
V
:__inference_Training_Data_Augmentation_layer_call_fn_12139

inputs
identityÛ
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
GPU 2J 8 *^
fYRW
U__inference_Training_Data_Augmentation_layer_call_and_return_conditional_losses_112482
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
Ø4
¦
S__inference_DoubleConv_LessDense_avg_layer_call_and_return_conditional_losses_11506$
 training_data_augmentation_input$
 training_data_augmentation_11291
conv2d_7_11316
conv2d_7_11318
conv2d_8_11374
conv2d_8_11376
dense_15_11446
dense_15_11448
dense_16_11473
dense_16_11475
dense_17_11500
dense_17_11502
identity¢2Training_Data_Augmentation/StatefulPartitionedCall¢ conv2d_7/StatefulPartitionedCall¢ conv2d_8/StatefulPartitionedCall¢ dense_15/StatefulPartitionedCall¢ dense_16/StatefulPartitionedCall¢ dense_17/StatefulPartitionedCall¢!dropout_7/StatefulPartitionedCall¢!dropout_8/StatefulPartitionedCallæ
2Training_Data_Augmentation/StatefulPartitionedCallStatefulPartitionedCall training_data_augmentation_input training_data_augmentation_11291*
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
GPU 2J 8 *^
fYRW
U__inference_Training_Data_Augmentation_layer_call_and_return_conditional_losses_1123424
2Training_Data_Augmentation/StatefulPartitionedCallÎ
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall;Training_Data_Augmentation/StatefulPartitionedCall:output:0conv2d_7_11316conv2d_7_11318*
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
GPU 2J 8 *L
fGRE
C__inference_conv2d_7_layer_call_and_return_conditional_losses_113052"
 conv2d_7/StatefulPartitionedCall
#average_pooling2d_2/PartitionedCallPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*
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
GPU 2J 8 *W
fRRP
N__inference_average_pooling2d_2_layer_call_and_return_conditional_losses_112572%
#average_pooling2d_2/PartitionedCall
!dropout_7/StatefulPartitionedCallStatefulPartitionedCall,average_pooling2d_2/PartitionedCall:output:0*
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
GPU 2J 8 *M
fHRF
D__inference_dropout_7_layer_call_and_return_conditional_losses_113342#
!dropout_7/StatefulPartitionedCall½
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCall*dropout_7/StatefulPartitionedCall:output:0conv2d_8_11374conv2d_8_11376*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_8_layer_call_and_return_conditional_losses_113632"
 conv2d_8/StatefulPartitionedCall
#average_pooling2d_3/PartitionedCallPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_average_pooling2d_3_layer_call_and_return_conditional_losses_112692%
#average_pooling2d_3/PartitionedCallÀ
!dropout_8/StatefulPartitionedCallStatefulPartitionedCall,average_pooling2d_3/PartitionedCall:output:0"^dropout_7/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_8_layer_call_and_return_conditional_losses_113922#
!dropout_8/StatefulPartitionedCallû
flatten_5/PartitionedCallPartitionedCall*dropout_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_flatten_5_layer_call_and_return_conditional_losses_114162
flatten_5/PartitionedCall®
 dense_15/StatefulPartitionedCallStatefulPartitionedCall"flatten_5/PartitionedCall:output:0dense_15_11446dense_15_11448*
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
GPU 2J 8 *L
fGRE
C__inference_dense_15_layer_call_and_return_conditional_losses_114352"
 dense_15/StatefulPartitionedCall´
 dense_16/StatefulPartitionedCallStatefulPartitionedCall)dense_15/StatefulPartitionedCall:output:0dense_16_11473dense_16_11475*
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
GPU 2J 8 *L
fGRE
C__inference_dense_16_layer_call_and_return_conditional_losses_114622"
 dense_16/StatefulPartitionedCall´
 dense_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0dense_17_11500dense_17_11502*
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
GPU 2J 8 *L
fGRE
C__inference_dense_17_layer_call_and_return_conditional_losses_114892"
 dense_17/StatefulPartitionedCall©
IdentityIdentity)dense_17/StatefulPartitionedCall:output:03^Training_Data_Augmentation/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall"^dropout_7/StatefulPartitionedCall"^dropout_8/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿ:::::::::::2h
2Training_Data_Augmentation/StatefulPartitionedCall2Training_Data_Augmentation/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2F
!dropout_7/StatefulPartitionedCall!dropout_7/StatefulPartitionedCall2F
!dropout_8/StatefulPartitionedCall!dropout_8/StatefulPartitionedCall:q m
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:
_user_specified_name" Training_Data_Augmentation_input
®
d
+__inference_random_flip_layer_call_fn_12365

inputs
identity¢StatefulPartitionedCallÿ
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
GPU 2J 8 *O
fJRH
F__inference_random_flip_layer_call_and_return_conditional_losses_109982
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
¼
`
D__inference_flatten_5_layer_call_and_return_conditional_losses_11416

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ 	  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
±
«
C__inference_dense_15_layer_call_and_return_conditional_losses_12255

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
:ÿÿÿÿÿÿÿÿÿ:::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ô
f
J__inference_random_rotation_layer_call_and_return_conditional_losses_11167

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


:__inference_Training_Data_Augmentation_layer_call_fn_11239
random_flip_input
unknown
identity¢StatefulPartitionedCall
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
GPU 2J 8 *^
fYRW
U__inference_Training_Data_Augmentation_layer_call_and_return_conditional_losses_112342
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
(
e
F__inference_random_flip_layer_call_and_return_conditional_losses_11041

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
ç
b
D__inference_dropout_7_layer_call_and_return_conditional_losses_12176

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
ü

:__inference_Training_Data_Augmentation_layer_call_fn_12134

inputs
unknown
identity¢StatefulPartitionedCallý
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
GPU 2J 8 *^
fYRW
U__inference_Training_Data_Augmentation_layer_call_and_return_conditional_losses_112342
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
ú
}
(__inference_conv2d_7_layer_call_fn_12159

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallû
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
GPU 2J 8 *L
fGRE
C__inference_conv2d_7_layer_call_and_return_conditional_losses_113052
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
«
«
C__inference_dense_16_layer_call_and_return_conditional_losses_11462

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
¹
i
J__inference_random_contrast_layer_call_and_return_conditional_losses_11197

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
ô
f
J__inference_random_rotation_layer_call_and_return_conditional_losses_12522

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
õ
a
:__inference_Training_Data_Augmentation_layer_call_fn_11251
random_flip_input
identityæ
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
GPU 2J 8 *^
fYRW
U__inference_Training_Data_Augmentation_layer_call_and_return_conditional_losses_112482
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
¯
q
U__inference_Training_Data_Augmentation_layer_call_and_return_conditional_losses_11248

inputs
identityä
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
GPU 2J 8 *O
fJRH
F__inference_random_flip_layer_call_and_return_conditional_losses_110452
random_flip/PartitionedCall
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
GPU 2J 8 *S
fNRL
J__inference_random_rotation_layer_call_and_return_conditional_losses_111672!
random_rotation/PartitionedCall
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
GPU 2J 8 *S
fNRL
J__inference_random_contrast_layer_call_and_return_conditional_losses_112012!
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
%

__inference__traced_save_12630
file_prefix.
*savev2_conv2d_7_kernel_read_readvariableop,
(savev2_conv2d_7_bias_read_readvariableop.
*savev2_conv2d_8_kernel_read_readvariableop,
(savev2_conv2d_8_bias_read_readvariableop.
*savev2_dense_15_kernel_read_readvariableop,
(savev2_dense_15_bias_read_readvariableop.
*savev2_dense_16_kernel_read_readvariableop,
(savev2_dense_16_bias_read_readvariableop.
*savev2_dense_17_kernel_read_readvariableop,
(savev2_dense_17_bias_read_readvariableop'
#savev2_variable_read_readvariableop	)
%savev2_variable_1_read_readvariableop	
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
value3B1 B+_temp_7889b41230cd4558afcfc0d8ddc69546/part2	
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
ShardedFilename¿
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ñ
valueÇBÄB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB:layer-0/layer-0/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUEB:layer-0/layer-1/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names¢
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*-
value$B"B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesÀ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_conv2d_7_kernel_read_readvariableop(savev2_conv2d_7_bias_read_readvariableop*savev2_conv2d_8_kernel_read_readvariableop(savev2_conv2d_8_bias_read_readvariableop*savev2_dense_15_kernel_read_readvariableop(savev2_dense_15_bias_read_readvariableop*savev2_dense_16_kernel_read_readvariableop(savev2_dense_16_bias_read_readvariableop*savev2_dense_17_kernel_read_readvariableop(savev2_dense_17_bias_read_readvariableop#savev2_variable_read_readvariableop%savev2_variable_1_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2		2
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

identity_1Identity_1:output:0*
_input_shapesv
t: :@:@:@@:@:
::	@:@:@
:
::: 2(
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
: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: 
`
ä
F__inference_random_flip_layer_call_and_return_conditional_losses_12356

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
`
ä
F__inference_random_flip_layer_call_and_return_conditional_losses_10998

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
Ê
h
/__inference_random_contrast_layer_call_fn_12556

inputs
identity¢StatefulPartitionedCallè
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
GPU 2J 8 *S
fNRL
J__inference_random_contrast_layer_call_and_return_conditional_losses_111972
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
	
«
C__inference_conv2d_8_layer_call_and_return_conditional_losses_11363

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
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
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@:::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
å
u
/__inference_random_rotation_layer_call_fn_12529

inputs
unknown
identity¢StatefulPartitionedCallò
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
GPU 2J 8 *S
fNRL
J__inference_random_rotation_layer_call_and_return_conditional_losses_111632
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
¤
E
)__inference_flatten_5_layer_call_fn_12244

inputs
identityÃ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_flatten_5_layer_call_and_return_conditional_losses_114162
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
°
»
J__inference_random_rotation_layer_call_and_return_conditional_losses_11163

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
ÿ
q
U__inference_Training_Data_Augmentation_layer_call_and_return_conditional_losses_12127

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
Á
b
F__inference_random_flip_layer_call_and_return_conditional_losses_11007

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
³
O
3__inference_average_pooling2d_3_layer_call_fn_11275

inputs
identityï
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
GPU 2J 8 *W
fRRP
N__inference_average_pooling2d_3_layer_call_and_return_conditional_losses_112692
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
Ð
|
U__inference_Training_Data_Augmentation_layer_call_and_return_conditional_losses_11222
random_flip_input
identityï
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
GPU 2J 8 *O
fJRH
F__inference_random_flip_layer_call_and_return_conditional_losses_110452
random_flip/PartitionedCall
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
GPU 2J 8 *S
fNRL
J__inference_random_rotation_layer_call_and_return_conditional_losses_111672!
random_rotation/PartitionedCall
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
GPU 2J 8 *S
fNRL
J__inference_random_contrast_layer_call_and_return_conditional_losses_112012!
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
²
E
)__inference_dropout_8_layer_call_fn_12233

inputs
identityÊ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_8_layer_call_and_return_conditional_losses_113972
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ô
f
J__inference_random_contrast_layer_call_and_return_conditional_losses_12551

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
°
«
C__inference_dense_17_layer_call_and_return_conditional_losses_11489

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
õ.
¹
S__inference_DoubleConv_LessDense_avg_layer_call_and_return_conditional_losses_11932

inputs+
'conv2d_7_conv2d_readvariableop_resource,
(conv2d_7_biasadd_readvariableop_resource+
'conv2d_8_conv2d_readvariableop_resource,
(conv2d_8_biasadd_readvariableop_resource+
'dense_15_matmul_readvariableop_resource,
(dense_15_biasadd_readvariableop_resource+
'dense_16_matmul_readvariableop_resource,
(dense_16_biasadd_readvariableop_resource+
'dense_17_matmul_readvariableop_resource,
(dense_17_biasadd_readvariableop_resource
identity°
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02 
conv2d_7/Conv2D/ReadVariableOp¿
conv2d_7/Conv2DConv2Dinputs&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
conv2d_7/Conv2D§
conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_7/BiasAdd/ReadVariableOp¬
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d_7/BiasAdd{
conv2d_7/ReluReluconv2d_7/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d_7/ReluØ
average_pooling2d_2/AvgPoolAvgPoolconv2d_7/Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
2
average_pooling2d_2/AvgPool
dropout_7/IdentityIdentity$average_pooling2d_2/AvgPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_7/Identity°
conv2d_8/Conv2D/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_8/Conv2D/ReadVariableOpÔ
conv2d_8/Conv2DConv2Ddropout_7/Identity:output:0&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
conv2d_8/Conv2D§
conv2d_8/BiasAdd/ReadVariableOpReadVariableOp(conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_8/BiasAdd/ReadVariableOp¬
conv2d_8/BiasAddBiasAddconv2d_8/Conv2D:output:0'conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d_8/BiasAdd{
conv2d_8/ReluReluconv2d_8/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d_8/ReluØ
average_pooling2d_3/AvgPoolAvgPoolconv2d_8/Relu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
2
average_pooling2d_3/AvgPool
dropout_8/IdentityIdentity$average_pooling2d_3/AvgPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_8/Identitys
flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ 	  2
flatten_5/Const
flatten_5/ReshapeReshapedropout_8/Identity:output:0flatten_5/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
flatten_5/Reshapeª
dense_15/MatMul/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02 
dense_15/MatMul/ReadVariableOp£
dense_15/MatMulMatMulflatten_5/Reshape:output:0&dense_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_15/MatMul¨
dense_15/BiasAdd/ReadVariableOpReadVariableOp(dense_15_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_15/BiasAdd/ReadVariableOp¦
dense_15/BiasAddBiasAdddense_15/MatMul:product:0'dense_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_15/BiasAddt
dense_15/ReluReludense_15/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_15/Relu©
dense_16/MatMul/ReadVariableOpReadVariableOp'dense_16_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02 
dense_16/MatMul/ReadVariableOp£
dense_16/MatMulMatMuldense_15/Relu:activations:0&dense_16/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_16/MatMul§
dense_16/BiasAdd/ReadVariableOpReadVariableOp(dense_16_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_16/BiasAdd/ReadVariableOp¥
dense_16/BiasAddBiasAdddense_16/MatMul:product:0'dense_16/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_16/BiasAdds
dense_16/ReluReludense_16/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_16/Relu¨
dense_17/MatMul/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource*
_output_shapes

:@
*
dtype02 
dense_17/MatMul/ReadVariableOp£
dense_17/MatMulMatMuldense_16/Relu:activations:0&dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_17/MatMul§
dense_17/BiasAdd/ReadVariableOpReadVariableOp(dense_17_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_17/BiasAdd/ReadVariableOp¥
dense_17/BiasAddBiasAdddense_17/MatMul:product:0'dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_17/BiasAdd|
dense_17/SoftmaxSoftmaxdense_17/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_17/Softmaxn
IdentityIdentitydense_17/Softmax:softmax:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿ:::::::::::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
°
»
J__inference_random_rotation_layer_call_and_return_conditional_losses_12518

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
ð
b
F__inference_random_flip_layer_call_and_return_conditional_losses_12402

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
Â
c
D__inference_dropout_8_layer_call_and_return_conditional_losses_12218

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
:ÿÿÿÿÿÿÿÿÿ@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¼
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
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
:ÿÿÿÿÿÿÿÿÿ@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¾
b
)__inference_dropout_7_layer_call_fn_12181

inputs
identity¢StatefulPartitionedCallâ
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
GPU 2J 8 *M
fHRF
D__inference_dropout_7_layer_call_and_return_conditional_losses_113342
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
ð
b
F__inference_random_flip_layer_call_and_return_conditional_losses_11045

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
³
O
3__inference_average_pooling2d_2_layer_call_fn_11263

inputs
identityï
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
GPU 2J 8 *W
fRRP
N__inference_average_pooling2d_2_layer_call_and_return_conditional_losses_112572
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
£
G
+__inference_random_flip_layer_call_fn_12370

inputs
identityç
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
GPU 2J 8 *O
fJRH
F__inference_random_flip_layer_call_and_return_conditional_losses_110072
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
Â
c
D__inference_dropout_7_layer_call_and_return_conditional_losses_12171

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

j
N__inference_average_pooling2d_2_layer_call_and_return_conditional_losses_11257

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
Â
d
+__inference_random_flip_layer_call_fn_12407

inputs
identity¢StatefulPartitionedCallä
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
GPU 2J 8 *O
fJRH
F__inference_random_flip_layer_call_and_return_conditional_losses_110412
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
4

S__inference_DoubleConv_LessDense_avg_layer_call_and_return_conditional_losses_11581

inputs$
 training_data_augmentation_11547
conv2d_7_11550
conv2d_7_11552
conv2d_8_11557
conv2d_8_11559
dense_15_11565
dense_15_11567
dense_16_11570
dense_16_11572
dense_17_11575
dense_17_11577
identity¢2Training_Data_Augmentation/StatefulPartitionedCall¢ conv2d_7/StatefulPartitionedCall¢ conv2d_8/StatefulPartitionedCall¢ dense_15/StatefulPartitionedCall¢ dense_16/StatefulPartitionedCall¢ dense_17/StatefulPartitionedCall¢!dropout_7/StatefulPartitionedCall¢!dropout_8/StatefulPartitionedCallÌ
2Training_Data_Augmentation/StatefulPartitionedCallStatefulPartitionedCallinputs training_data_augmentation_11547*
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
GPU 2J 8 *^
fYRW
U__inference_Training_Data_Augmentation_layer_call_and_return_conditional_losses_1123424
2Training_Data_Augmentation/StatefulPartitionedCallÎ
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall;Training_Data_Augmentation/StatefulPartitionedCall:output:0conv2d_7_11550conv2d_7_11552*
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
GPU 2J 8 *L
fGRE
C__inference_conv2d_7_layer_call_and_return_conditional_losses_113052"
 conv2d_7/StatefulPartitionedCall
#average_pooling2d_2/PartitionedCallPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*
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
GPU 2J 8 *W
fRRP
N__inference_average_pooling2d_2_layer_call_and_return_conditional_losses_112572%
#average_pooling2d_2/PartitionedCall
!dropout_7/StatefulPartitionedCallStatefulPartitionedCall,average_pooling2d_2/PartitionedCall:output:0*
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
GPU 2J 8 *M
fHRF
D__inference_dropout_7_layer_call_and_return_conditional_losses_113342#
!dropout_7/StatefulPartitionedCall½
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCall*dropout_7/StatefulPartitionedCall:output:0conv2d_8_11557conv2d_8_11559*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_8_layer_call_and_return_conditional_losses_113632"
 conv2d_8/StatefulPartitionedCall
#average_pooling2d_3/PartitionedCallPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_average_pooling2d_3_layer_call_and_return_conditional_losses_112692%
#average_pooling2d_3/PartitionedCallÀ
!dropout_8/StatefulPartitionedCallStatefulPartitionedCall,average_pooling2d_3/PartitionedCall:output:0"^dropout_7/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_8_layer_call_and_return_conditional_losses_113922#
!dropout_8/StatefulPartitionedCallû
flatten_5/PartitionedCallPartitionedCall*dropout_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_flatten_5_layer_call_and_return_conditional_losses_114162
flatten_5/PartitionedCall®
 dense_15/StatefulPartitionedCallStatefulPartitionedCall"flatten_5/PartitionedCall:output:0dense_15_11565dense_15_11567*
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
GPU 2J 8 *L
fGRE
C__inference_dense_15_layer_call_and_return_conditional_losses_114352"
 dense_15/StatefulPartitionedCall´
 dense_16/StatefulPartitionedCallStatefulPartitionedCall)dense_15/StatefulPartitionedCall:output:0dense_16_11570dense_16_11572*
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
GPU 2J 8 *L
fGRE
C__inference_dense_16_layer_call_and_return_conditional_losses_114622"
 dense_16/StatefulPartitionedCall´
 dense_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0dense_17_11575dense_17_11577*
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
GPU 2J 8 *L
fGRE
C__inference_dense_17_layer_call_and_return_conditional_losses_114892"
 dense_17/StatefulPartitionedCall©
IdentityIdentity)dense_17/StatefulPartitionedCall:output:03^Training_Data_Augmentation/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall"^dropout_7/StatefulPartitionedCall"^dropout_8/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:ÿÿÿÿÿÿÿÿÿ:::::::::::2h
2Training_Data_Augmentation/StatefulPartitionedCall2Training_Data_Augmentation/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall2F
!dropout_7/StatefulPartitionedCall!dropout_7/StatefulPartitionedCall2F
!dropout_8/StatefulPartitionedCall!dropout_8/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
±
«
C__inference_dense_15_layer_call_and_return_conditional_losses_11435

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
:ÿÿÿÿÿÿÿÿÿ:::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	

8__inference_DoubleConv_LessDense_avg_layer_call_fn_11984

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
identity¢StatefulPartitionedCallë
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8 *\
fWRU
S__inference_DoubleConv_LessDense_avg_layer_call_and_return_conditional_losses_116432
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿ::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¶
G
+__inference_random_flip_layer_call_fn_12412

inputs
identityÌ
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
GPU 2J 8 *O
fJRH
F__inference_random_flip_layer_call_and_return_conditional_losses_110452
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
ë
æ
U__inference_Training_Data_Augmentation_layer_call_and_return_conditional_losses_12123

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
	
«
C__inference_conv2d_7_layer_call_and_return_conditional_losses_12150

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
«
«
C__inference_dense_16_layer_call_and_return_conditional_losses_12275

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
¾
K
/__inference_random_rotation_layer_call_fn_12534

inputs
identityÐ
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
GPU 2J 8 *S
fNRL
J__inference_random_rotation_layer_call_and_return_conditional_losses_111672
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
Å.
é
S__inference_DoubleConv_LessDense_avg_layer_call_and_return_conditional_losses_11643

inputs
conv2d_7_11612
conv2d_7_11614
conv2d_8_11619
conv2d_8_11621
dense_15_11627
dense_15_11629
dense_16_11632
dense_16_11634
dense_17_11637
dense_17_11639
identity¢ conv2d_7/StatefulPartitionedCall¢ conv2d_8/StatefulPartitionedCall¢ dense_15/StatefulPartitionedCall¢ dense_16/StatefulPartitionedCall¢ dense_17/StatefulPartitionedCall
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
GPU 2J 8 *^
fYRW
U__inference_Training_Data_Augmentation_layer_call_and_return_conditional_losses_112482,
*Training_Data_Augmentation/PartitionedCallÆ
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall3Training_Data_Augmentation/PartitionedCall:output:0conv2d_7_11612conv2d_7_11614*
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
GPU 2J 8 *L
fGRE
C__inference_conv2d_7_layer_call_and_return_conditional_losses_113052"
 conv2d_7/StatefulPartitionedCall
#average_pooling2d_2/PartitionedCallPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*
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
GPU 2J 8 *W
fRRP
N__inference_average_pooling2d_2_layer_call_and_return_conditional_losses_112572%
#average_pooling2d_2/PartitionedCall
dropout_7/PartitionedCallPartitionedCall,average_pooling2d_2/PartitionedCall:output:0*
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
GPU 2J 8 *M
fHRF
D__inference_dropout_7_layer_call_and_return_conditional_losses_113392
dropout_7/PartitionedCallµ
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCall"dropout_7/PartitionedCall:output:0conv2d_8_11619conv2d_8_11621*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_conv2d_8_layer_call_and_return_conditional_losses_113632"
 conv2d_8/StatefulPartitionedCall
#average_pooling2d_3/PartitionedCallPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_average_pooling2d_3_layer_call_and_return_conditional_losses_112692%
#average_pooling2d_3/PartitionedCall
dropout_8/PartitionedCallPartitionedCall,average_pooling2d_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_8_layer_call_and_return_conditional_losses_113972
dropout_8/PartitionedCalló
flatten_5/PartitionedCallPartitionedCall"dropout_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_flatten_5_layer_call_and_return_conditional_losses_114162
flatten_5/PartitionedCall®
 dense_15/StatefulPartitionedCallStatefulPartitionedCall"flatten_5/PartitionedCall:output:0dense_15_11627dense_15_11629*
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
GPU 2J 8 *L
fGRE
C__inference_dense_15_layer_call_and_return_conditional_losses_114352"
 dense_15/StatefulPartitionedCall´
 dense_16/StatefulPartitionedCallStatefulPartitionedCall)dense_15/StatefulPartitionedCall:output:0dense_16_11632dense_16_11634*
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
GPU 2J 8 *L
fGRE
C__inference_dense_16_layer_call_and_return_conditional_losses_114622"
 dense_16/StatefulPartitionedCall´
 dense_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0dense_17_11637dense_17_11639*
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
GPU 2J 8 *L
fGRE
C__inference_dense_17_layer_call_and_return_conditional_losses_114892"
 dense_17/StatefulPartitionedCall¬
IdentityIdentity)dense_17/StatefulPartitionedCall:output:0!^conv2d_7/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall!^dense_16/StatefulPartitionedCall!^dense_17/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:ÿÿÿÿÿÿÿÿÿ::::::::::2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ø

U__inference_Training_Data_Augmentation_layer_call_and_return_conditional_losses_11234

inputs
random_rotation_11229
identity¢'random_contrast/StatefulPartitionedCall¢#random_flip/StatefulPartitionedCall¢'random_rotation/StatefulPartitionedCallü
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
GPU 2J 8 *O
fJRH
F__inference_random_flip_layer_call_and_return_conditional_losses_110412%
#random_flip/StatefulPartitionedCallÆ
'random_rotation/StatefulPartitionedCallStatefulPartitionedCall,random_flip/StatefulPartitionedCall:output:0random_rotation_11229*
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
GPU 2J 8 *S
fNRL
J__inference_random_rotation_layer_call_and_return_conditional_losses_111632)
'random_rotation/StatefulPartitionedCallØ
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
GPU 2J 8 *S
fNRL
J__inference_random_contrast_layer_call_and_return_conditional_losses_111972)
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
 
_user_specified_nameinputs"¸L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*å
serving_defaultÑ
u
 Training_Data_Augmentation_inputQ
2serving_default_Training_Data_Augmentation_input:0ÿÿÿÿÿÿÿÿÿ<
dense_170
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿ
tensorflow/serving/predict:¹¤
V
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
¦__call__
§_default_save_signature
+¨&call_and_return_all_conditional_losses"¯R
_tf_keras_sequentialR{"class_name": "Sequential", "name": "DoubleConv_LessDense_avg", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "DoubleConv_LessDense_avg", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Training_Data_Augmentation_input"}}, {"class_name": "Sequential", "config": {"name": "Training_Data_Augmentation", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "random_flip_input"}}, {"class_name": "RandomFlip", "config": {"name": "random_flip", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "mode": "horizontal", "seed": null}}, {"class_name": "RandomRotation", "config": {"name": "random_rotation", "trainable": true, "dtype": "float32", "factor": 0.2, "fill_mode": "reflect", "interpolation": "bilinear", "seed": null}}, {"class_name": "RandomContrast", "config": {"name": "random_contrast", "trainable": true, "dtype": "float32", "factor": 0.1, "seed": null}}]}}, {"class_name": "Conv2D", "config": {"name": "conv2d_7", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "AveragePooling2D", "config": {"name": "average_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_7", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_8", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "AveragePooling2D", "config": {"name": "average_pooling2d_3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_8", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten_5", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_15", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_16", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_17", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "DoubleConv_LessDense_avg", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "Training_Data_Augmentation_input"}}, {"class_name": "Sequential", "config": {"name": "Training_Data_Augmentation", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "random_flip_input"}}, {"class_name": "RandomFlip", "config": {"name": "random_flip", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "mode": "horizontal", "seed": null}}, {"class_name": "RandomRotation", "config": {"name": "random_rotation", "trainable": true, "dtype": "float32", "factor": 0.2, "fill_mode": "reflect", "interpolation": "bilinear", "seed": null}}, {"class_name": "RandomContrast", "config": {"name": "random_contrast", "trainable": true, "dtype": "float32", "factor": 0.1, "seed": null}}]}}, {"class_name": "Conv2D", "config": {"name": "conv2d_7", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "AveragePooling2D", "config": {"name": "average_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_7", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}}, {"class_name": "Conv2D", "config": {"name": "conv2d_8", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "AveragePooling2D", "config": {"name": "average_pooling2d_3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_8", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten_5", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_15", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_16", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_17", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "sparse_categorical_crossentropy", "metrics": ["accuracy"], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.01, "decay": 0.0, "beta_1": 0.9, "beta_2": 0.999, "epsilon": 1e-07, "amsgrad": false}}}}
Û
layer-0
layer-1
layer-2
regularization_losses
	variables
trainable_variables
	keras_api
©__call__
+ª&call_and_return_all_conditional_losses"£
_tf_keras_sequential{"class_name": "Sequential", "name": "Training_Data_Augmentation", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "Training_Data_Augmentation", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "random_flip_input"}}, {"class_name": "RandomFlip", "config": {"name": "random_flip", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "mode": "horizontal", "seed": null}}, {"class_name": "RandomRotation", "config": {"name": "random_rotation", "trainable": true, "dtype": "float32", "factor": 0.2, "fill_mode": "reflect", "interpolation": "bilinear", "seed": null}}, {"class_name": "RandomContrast", "config": {"name": "random_contrast", "trainable": true, "dtype": "float32", "factor": 0.1, "seed": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "Training_Data_Augmentation", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "random_flip_input"}}, {"class_name": "RandomFlip", "config": {"name": "random_flip", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "mode": "horizontal", "seed": null}}, {"class_name": "RandomRotation", "config": {"name": "random_rotation", "trainable": true, "dtype": "float32", "factor": 0.2, "fill_mode": "reflect", "interpolation": "bilinear", "seed": null}}, {"class_name": "RandomContrast", "config": {"name": "random_contrast", "trainable": true, "dtype": "float32", "factor": 0.1, "seed": null}}]}}}
ô


kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
«__call__
+¬&call_and_return_all_conditional_losses"Í	
_tf_keras_layer³	{"class_name": "Conv2D", "name": "conv2d_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_7", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 1]}}

regularization_losses
 	variables
!trainable_variables
"	keras_api
­__call__
+®&call_and_return_all_conditional_losses"ü
_tf_keras_layerâ{"class_name": "AveragePooling2D", "name": "average_pooling2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "average_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ç
#regularization_losses
$	variables
%trainable_variables
&	keras_api
¯__call__
+°&call_and_return_all_conditional_losses"Ö
_tf_keras_layer¼{"class_name": "Dropout", "name": "dropout_7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_7", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}}
õ	

'kernel
(bias
)regularization_losses
*	variables
+trainable_variables
,	keras_api
±__call__
+²&call_and_return_all_conditional_losses"Î
_tf_keras_layer´{"class_name": "Conv2D", "name": "conv2d_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_8", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [2, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 13, 13, 64]}}

-regularization_losses
.	variables
/trainable_variables
0	keras_api
³__call__
+´&call_and_return_all_conditional_losses"ü
_tf_keras_layerâ{"class_name": "AveragePooling2D", "name": "average_pooling2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "average_pooling2d_3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ç
1regularization_losses
2	variables
3trainable_variables
4	keras_api
µ__call__
+¶&call_and_return_all_conditional_losses"Ö
_tf_keras_layer¼{"class_name": "Dropout", "name": "dropout_8", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_8", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}}
è
5regularization_losses
6	variables
7trainable_variables
8	keras_api
·__call__
+¸&call_and_return_all_conditional_losses"×
_tf_keras_layer½{"class_name": "Flatten", "name": "flatten_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_5", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
ù

9kernel
:bias
;regularization_losses
<	variables
=trainable_variables
>	keras_api
¹__call__
+º&call_and_return_all_conditional_losses"Ò
_tf_keras_layer¸{"class_name": "Dense", "name": "dense_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_15", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2304}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2304]}}
ö

?kernel
@bias
Aregularization_losses
B	variables
Ctrainable_variables
D	keras_api
»__call__
+¼&call_and_return_all_conditional_losses"Ï
_tf_keras_layerµ{"class_name": "Dense", "name": "dense_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_16", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
÷

Ekernel
Fbias
Gregularization_losses
H	variables
Itrainable_variables
J	keras_api
½__call__
+¾&call_and_return_all_conditional_losses"Ð
_tf_keras_layer¶{"class_name": "Dense", "name": "dense_17", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_17", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
"
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
Î

Klayers
Llayer_metrics
Mnon_trainable_variables
regularization_losses
	variables
trainable_variables
Nmetrics
Olayer_regularization_losses
¦__call__
§_default_save_signature
+¨&call_and_return_all_conditional_losses
'¨"call_and_return_conditional_losses"
_generic_user_object
-
¿serving_default"
signature_map
ý
P_rng
Qregularization_losses
R	variables
Strainable_variables
T	keras_api
À__call__
+Á&call_and_return_all_conditional_losses"â
_tf_keras_layerÈ{"class_name": "RandomFlip", "name": "random_flip", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "random_flip", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "mode": "horizontal", "seed": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
¦
U_rng
Vregularization_losses
W	variables
Xtrainable_variables
Y	keras_api
Â__call__
+Ã&call_and_return_all_conditional_losses"
_tf_keras_layerñ{"class_name": "RandomRotation", "name": "random_rotation", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "random_rotation", "trainable": true, "dtype": "float32", "factor": 0.2, "fill_mode": "reflect", "interpolation": "bilinear", "seed": null}}
ç
Zregularization_losses
[	variables
\trainable_variables
]	keras_api
Ä__call__
+Å&call_and_return_all_conditional_losses"Ö
_tf_keras_layer¼{"class_name": "RandomContrast", "name": "random_contrast", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "random_contrast", "trainable": true, "dtype": "float32", "factor": 0.1, "seed": null}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°

^layers
_layer_metrics
`non_trainable_variables
regularization_losses
	variables
trainable_variables
ametrics
blayer_regularization_losses
©__call__
+ª&call_and_return_all_conditional_losses
'ª"call_and_return_conditional_losses"
_generic_user_object
):'@2conv2d_7/kernel
:@2conv2d_7/bias
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

clayers
dlayer_metrics
enon_trainable_variables
regularization_losses
	variables
trainable_variables
fmetrics
glayer_regularization_losses
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

hlayers
ilayer_metrics
jnon_trainable_variables
regularization_losses
 	variables
!trainable_variables
kmetrics
llayer_regularization_losses
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

mlayers
nlayer_metrics
onon_trainable_variables
#regularization_losses
$	variables
%trainable_variables
pmetrics
qlayer_regularization_losses
¯__call__
+°&call_and_return_all_conditional_losses
'°"call_and_return_conditional_losses"
_generic_user_object
):'@@2conv2d_8/kernel
:@2conv2d_8/bias
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

rlayers
slayer_metrics
tnon_trainable_variables
)regularization_losses
*	variables
+trainable_variables
umetrics
vlayer_regularization_losses
±__call__
+²&call_and_return_all_conditional_losses
'²"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°

wlayers
xlayer_metrics
ynon_trainable_variables
-regularization_losses
.	variables
/trainable_variables
zmetrics
{layer_regularization_losses
³__call__
+´&call_and_return_all_conditional_losses
'´"call_and_return_conditional_losses"
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
1regularization_losses
2	variables
3trainable_variables
metrics
 layer_regularization_losses
µ__call__
+¶&call_and_return_all_conditional_losses
'¶"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
layers
layer_metrics
non_trainable_variables
5regularization_losses
6	variables
7trainable_variables
metrics
 layer_regularization_losses
·__call__
+¸&call_and_return_all_conditional_losses
'¸"call_and_return_conditional_losses"
_generic_user_object
#:!
2dense_15/kernel
:2dense_15/bias
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
µ
layers
layer_metrics
non_trainable_variables
;regularization_losses
<	variables
=trainable_variables
metrics
 layer_regularization_losses
¹__call__
+º&call_and_return_all_conditional_losses
'º"call_and_return_conditional_losses"
_generic_user_object
": 	@2dense_16/kernel
:@2dense_16/bias
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
µ
layers
layer_metrics
non_trainable_variables
Aregularization_losses
B	variables
Ctrainable_variables
metrics
 layer_regularization_losses
»__call__
+¼&call_and_return_all_conditional_losses
'¼"call_and_return_conditional_losses"
_generic_user_object
!:@
2dense_17/kernel
:
2dense_17/bias
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
µ
layers
layer_metrics
non_trainable_variables
Gregularization_losses
H	variables
Itrainable_variables
metrics
 layer_regularization_losses
½__call__
+¾&call_and_return_all_conditional_losses
'¾"call_and_return_conditional_losses"
_generic_user_object
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
/

_state_var"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
layers
layer_metrics
non_trainable_variables
Qregularization_losses
R	variables
Strainable_variables
metrics
 layer_regularization_losses
À__call__
+Á&call_and_return_all_conditional_losses
'Á"call_and_return_conditional_losses"
_generic_user_object
/

_state_var"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
layers
layer_metrics
non_trainable_variables
Vregularization_losses
W	variables
Xtrainable_variables
metrics
  layer_regularization_losses
Â__call__
+Ã&call_and_return_all_conditional_losses
'Ã"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
¡layers
¢layer_metrics
£non_trainable_variables
Zregularization_losses
[	variables
\trainable_variables
¤metrics
 ¥layer_regularization_losses
Ä__call__
+Å&call_and_return_all_conditional_losses
'Å"call_and_return_conditional_losses"
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
®2«
8__inference_DoubleConv_LessDense_avg_layer_call_fn_11959
8__inference_DoubleConv_LessDense_avg_layer_call_fn_11666
8__inference_DoubleConv_LessDense_avg_layer_call_fn_11984
8__inference_DoubleConv_LessDense_avg_layer_call_fn_11606À
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
ÿ2ü
 __inference__wrapped_model_10886×
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
2
S__inference_DoubleConv_LessDense_avg_layer_call_and_return_conditional_losses_11932
S__inference_DoubleConv_LessDense_avg_layer_call_and_return_conditional_losses_11506
S__inference_DoubleConv_LessDense_avg_layer_call_and_return_conditional_losses_11887
S__inference_DoubleConv_LessDense_avg_layer_call_and_return_conditional_losses_11541À
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
¶2³
:__inference_Training_Data_Augmentation_layer_call_fn_12134
:__inference_Training_Data_Augmentation_layer_call_fn_11251
:__inference_Training_Data_Augmentation_layer_call_fn_12139
:__inference_Training_Data_Augmentation_layer_call_fn_11239À
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
¢2
U__inference_Training_Data_Augmentation_layer_call_and_return_conditional_losses_11215
U__inference_Training_Data_Augmentation_layer_call_and_return_conditional_losses_12127
U__inference_Training_Data_Augmentation_layer_call_and_return_conditional_losses_11222
U__inference_Training_Data_Augmentation_layer_call_and_return_conditional_losses_12123À
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
Ò2Ï
(__inference_conv2d_7_layer_call_fn_12159¢
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
í2ê
C__inference_conv2d_7_layer_call_and_return_conditional_losses_12150¢
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
2
3__inference_average_pooling2d_2_layer_call_fn_11263à
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
¶2³
N__inference_average_pooling2d_2_layer_call_and_return_conditional_losses_11257à
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
2
)__inference_dropout_7_layer_call_fn_12181
)__inference_dropout_7_layer_call_fn_12186´
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
Æ2Ã
D__inference_dropout_7_layer_call_and_return_conditional_losses_12176
D__inference_dropout_7_layer_call_and_return_conditional_losses_12171´
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
Ò2Ï
(__inference_conv2d_8_layer_call_fn_12206¢
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
í2ê
C__inference_conv2d_8_layer_call_and_return_conditional_losses_12197¢
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
2
3__inference_average_pooling2d_3_layer_call_fn_11275à
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
¶2³
N__inference_average_pooling2d_3_layer_call_and_return_conditional_losses_11269à
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
2
)__inference_dropout_8_layer_call_fn_12233
)__inference_dropout_8_layer_call_fn_12228´
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
Æ2Ã
D__inference_dropout_8_layer_call_and_return_conditional_losses_12223
D__inference_dropout_8_layer_call_and_return_conditional_losses_12218´
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
Ó2Ð
)__inference_flatten_5_layer_call_fn_12244¢
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
î2ë
D__inference_flatten_5_layer_call_and_return_conditional_losses_12239¢
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
Ò2Ï
(__inference_dense_15_layer_call_fn_12264¢
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
í2ê
C__inference_dense_15_layer_call_and_return_conditional_losses_12255¢
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
Ò2Ï
(__inference_dense_16_layer_call_fn_12284¢
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
í2ê
C__inference_dense_16_layer_call_and_return_conditional_losses_12275¢
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
Ò2Ï
(__inference_dense_17_layer_call_fn_12304¢
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
í2ê
C__inference_dense_17_layer_call_and_return_conditional_losses_12295¢
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
KBI
#__inference_signature_wrapper_11693 Training_Data_Augmentation_input
î2ë
+__inference_random_flip_layer_call_fn_12412
+__inference_random_flip_layer_call_fn_12365
+__inference_random_flip_layer_call_fn_12407
+__inference_random_flip_layer_call_fn_12370´
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
Ú2×
F__inference_random_flip_layer_call_and_return_conditional_losses_12356
F__inference_random_flip_layer_call_and_return_conditional_losses_12360
F__inference_random_flip_layer_call_and_return_conditional_losses_12398
F__inference_random_flip_layer_call_and_return_conditional_losses_12402´
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
2
/__inference_random_rotation_layer_call_fn_12534
/__inference_random_rotation_layer_call_fn_12529´
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
Ò2Ï
J__inference_random_rotation_layer_call_and_return_conditional_losses_12518
J__inference_random_rotation_layer_call_and_return_conditional_losses_12522´
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
2
/__inference_random_contrast_layer_call_fn_12561
/__inference_random_contrast_layer_call_fn_12556´
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
Ò2Ï
J__inference_random_contrast_layer_call_and_return_conditional_losses_12547
J__inference_random_contrast_layer_call_and_return_conditional_losses_12551´
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
 è
S__inference_DoubleConv_LessDense_avg_layer_call_and_return_conditional_losses_11506'(9:?@EFY¢V
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
 æ
S__inference_DoubleConv_LessDense_avg_layer_call_and_return_conditional_losses_11541
'(9:?@EFY¢V
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
 Í
S__inference_DoubleConv_LessDense_avg_layer_call_and_return_conditional_losses_11887v'(9:?@EF?¢<
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
 Ë
S__inference_DoubleConv_LessDense_avg_layer_call_and_return_conditional_losses_11932t
'(9:?@EF?¢<
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
 À
8__inference_DoubleConv_LessDense_avg_layer_call_fn_11606'(9:?@EFY¢V
O¢L
B?
 Training_Data_Augmentation_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
¾
8__inference_DoubleConv_LessDense_avg_layer_call_fn_11666
'(9:?@EFY¢V
O¢L
B?
 Training_Data_Augmentation_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
¥
8__inference_DoubleConv_LessDense_avg_layer_call_fn_11959i'(9:?@EF?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
£
8__inference_DoubleConv_LessDense_avg_layer_call_fn_11984g
'(9:?@EF?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
Ø
U__inference_Training_Data_Augmentation_layer_call_and_return_conditional_losses_11215J¢G
@¢=
30
random_flip_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 Ô
U__inference_Training_Data_Augmentation_layer_call_and_return_conditional_losses_11222{J¢G
@¢=
30
random_flip_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 Í
U__inference_Training_Data_Augmentation_layer_call_and_return_conditional_losses_12123t?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 É
U__inference_Training_Data_Augmentation_layer_call_and_return_conditional_losses_12127p?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 °
:__inference_Training_Data_Augmentation_layer_call_fn_11239rJ¢G
@¢=
30
random_flip_inputÿÿÿÿÿÿÿÿÿ
p

 
ª " ÿÿÿÿÿÿÿÿÿ¬
:__inference_Training_Data_Augmentation_layer_call_fn_11251nJ¢G
@¢=
30
random_flip_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª " ÿÿÿÿÿÿÿÿÿ¥
:__inference_Training_Data_Augmentation_layer_call_fn_12134g?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª " ÿÿÿÿÿÿÿÿÿ¡
:__inference_Training_Data_Augmentation_layer_call_fn_12139c?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª " ÿÿÿÿÿÿÿÿÿ¹
 __inference__wrapped_model_10886
'(9:?@EFQ¢N
G¢D
B?
 Training_Data_Augmentation_inputÿÿÿÿÿÿÿÿÿ
ª "3ª0
.
dense_17"
dense_17ÿÿÿÿÿÿÿÿÿ
ñ
N__inference_average_pooling2d_2_layer_call_and_return_conditional_losses_11257R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 É
3__inference_average_pooling2d_2_layer_call_fn_11263R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿñ
N__inference_average_pooling2d_3_layer_call_and_return_conditional_losses_11269R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 É
3__inference_average_pooling2d_3_layer_call_fn_11275R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ³
C__inference_conv2d_7_layer_call_and_return_conditional_losses_12150l7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 
(__inference_conv2d_7_layer_call_fn_12159_7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ
ª " ÿÿÿÿÿÿÿÿÿ@³
C__inference_conv2d_8_layer_call_and_return_conditional_losses_12197l'(7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 
(__inference_conv2d_8_layer_call_fn_12206_'(7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª " ÿÿÿÿÿÿÿÿÿ@¥
C__inference_dense_15_layer_call_and_return_conditional_losses_12255^9:0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 }
(__inference_dense_15_layer_call_fn_12264Q9:0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¤
C__inference_dense_16_layer_call_and_return_conditional_losses_12275]?@0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 |
(__inference_dense_16_layer_call_fn_12284P?@0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ@£
C__inference_dense_17_layer_call_and_return_conditional_losses_12295\EF/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 {
(__inference_dense_17_layer_call_fn_12304OEF/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ
´
D__inference_dropout_7_layer_call_and_return_conditional_losses_12171l;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 ´
D__inference_dropout_7_layer_call_and_return_conditional_losses_12176l;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 
)__inference_dropout_7_layer_call_fn_12181_;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p
ª " ÿÿÿÿÿÿÿÿÿ@
)__inference_dropout_7_layer_call_fn_12186_;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª " ÿÿÿÿÿÿÿÿÿ@´
D__inference_dropout_8_layer_call_and_return_conditional_losses_12218l;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 ´
D__inference_dropout_8_layer_call_and_return_conditional_losses_12223l;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 
)__inference_dropout_8_layer_call_fn_12228_;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p
ª " ÿÿÿÿÿÿÿÿÿ@
)__inference_dropout_8_layer_call_fn_12233_;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª " ÿÿÿÿÿÿÿÿÿ@©
D__inference_flatten_5_layer_call_and_return_conditional_losses_12239a7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
)__inference_flatten_5_layer_call_fn_12244T7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿº
J__inference_random_contrast_layer_call_and_return_conditional_losses_12547l;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 º
J__inference_random_contrast_layer_call_and_return_conditional_losses_12551l;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 
/__inference_random_contrast_layer_call_fn_12556_;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ
p
ª " ÿÿÿÿÿÿÿÿÿ
/__inference_random_contrast_layer_call_fn_12561_;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ
p 
ª " ÿÿÿÿÿÿÿÿÿí
F__inference_random_flip_layer_call_and_return_conditional_losses_12356¢V¢S
L¢I
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 í
F__inference_random_flip_layer_call_and_return_conditional_losses_12360¢V¢S
L¢I
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¶
F__inference_random_flip_layer_call_and_return_conditional_losses_12398l;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 ¶
F__inference_random_flip_layer_call_and_return_conditional_losses_12402l;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 Å
+__inference_random_flip_layer_call_fn_12365V¢S
L¢I
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÅ
+__inference_random_flip_layer_call_fn_12370V¢S
L¢I
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
+__inference_random_flip_layer_call_fn_12407_;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ
p
ª " ÿÿÿÿÿÿÿÿÿ
+__inference_random_flip_layer_call_fn_12412_;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ
p 
ª " ÿÿÿÿÿÿÿÿÿ¾
J__inference_random_rotation_layer_call_and_return_conditional_losses_12518p;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 º
J__inference_random_rotation_layer_call_and_return_conditional_losses_12522l;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ
 
/__inference_random_rotation_layer_call_fn_12529c;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ
p
ª " ÿÿÿÿÿÿÿÿÿ
/__inference_random_rotation_layer_call_fn_12534_;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ
p 
ª " ÿÿÿÿÿÿÿÿÿà
#__inference_signature_wrapper_11693¸
'(9:?@EFu¢r
¢ 
kªh
f
 Training_Data_Augmentation_inputB?
 Training_Data_Augmentation_inputÿÿÿÿÿÿÿÿÿ"3ª0
.
dense_17"
dense_17ÿÿÿÿÿÿÿÿÿ
