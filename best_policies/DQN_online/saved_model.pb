ы
О""
D
AddV2
x"T
y"T
z"T"
Ttype:
2	

ArgMax

input"T
	dimension"Tidx
output"output_type"!
Ttype:
2	
"
Tidxtype0:
2	"!
output_typetype0	:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 

BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
A
BroadcastArgs
s0"T
s1"T
r0"T"
Ttype0:
2	
Z
BroadcastTo

input"T
shape"Tidx
output"T"	
Ttype"
Tidxtype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resource
R
Equal
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
Ў
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 
>
Minimum
x"T
y"T
z"T"
Ttype:
2	
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
Г
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:

Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( ""
Ttype:
2	"
Tidxtype0:
2	
f
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx" 
Tidxtype0:
2
	
@
ReadVariableOp
resource
value"dtype"
dtypetype
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
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
A
SelectV2
	condition

t"T
e"T
output"T"	
Ttype
<
Selu
features"T
activations"T"
Ttype:
2
d
Shape

input"T&
output"out_typeэout_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
С
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
executor_typestring Ј
@
StaticRegexFullMatch	
input

output
"
patternstring
ї
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
-
Tanh
x"T
y"T"
Ttype:

2
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
А
VarHandleOp
resource"
	containerstring "
shared_namestring "

debug_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.15.02v2.15.0-rc1-8-g6887368d6d48щи
ц
)QRnnNetwork/num_action_project/dense/biasVarHandleOp*
_output_shapes
: *:

debug_name,*QRnnNetwork/num_action_project/dense/bias/*
dtype0*
shape:*:
shared_name+)QRnnNetwork/num_action_project/dense/bias
Ѓ
=QRnnNetwork/num_action_project/dense/bias/Read/ReadVariableOpReadVariableOp)QRnnNetwork/num_action_project/dense/bias*
_output_shapes
:*
dtype0
№
+QRnnNetwork/num_action_project/dense/kernelVarHandleOp*
_output_shapes
: *<

debug_name.,QRnnNetwork/num_action_project/dense/kernel/*
dtype0*
shape
:d*<
shared_name-+QRnnNetwork/num_action_project/dense/kernel
Ћ
?QRnnNetwork/num_action_project/dense/kernel/Read/ReadVariableOpReadVariableOp+QRnnNetwork/num_action_project/dense/kernel*
_output_shapes

:d*
dtype0
Г
QRnnNetwork/dense_3/biasVarHandleOp*
_output_shapes
: *)

debug_nameQRnnNetwork/dense_3/bias/*
dtype0*
shape:d*)
shared_nameQRnnNetwork/dense_3/bias

,QRnnNetwork/dense_3/bias/Read/ReadVariableOpReadVariableOpQRnnNetwork/dense_3/bias*
_output_shapes
:d*
dtype0
О
QRnnNetwork/dense_3/kernelVarHandleOp*
_output_shapes
: *+

debug_nameQRnnNetwork/dense_3/kernel/*
dtype0*
shape:	иd*+
shared_nameQRnnNetwork/dense_3/kernel

.QRnnNetwork/dense_3/kernel/Read/ReadVariableOpReadVariableOpQRnnNetwork/dense_3/kernel*
_output_shapes
:	иd*
dtype0
Д
QRnnNetwork/dense_2/biasVarHandleOp*
_output_shapes
: *)

debug_nameQRnnNetwork/dense_2/bias/*
dtype0*
shape:и*)
shared_nameQRnnNetwork/dense_2/bias

,QRnnNetwork/dense_2/bias/Read/ReadVariableOpReadVariableOpQRnnNetwork/dense_2/bias*
_output_shapes	
:и*
dtype0
О
QRnnNetwork/dense_2/kernelVarHandleOp*
_output_shapes
: *+

debug_nameQRnnNetwork/dense_2/kernel/*
dtype0*
shape:	Mи*+
shared_nameQRnnNetwork/dense_2/kernel

.QRnnNetwork/dense_2/kernel/Read/ReadVariableOpReadVariableOpQRnnNetwork/dense_2/kernel*
_output_shapes
:	Mи*
dtype0
Щ
QRnnNetwork/dynamic_unroll/biasVarHandleOp*
_output_shapes
: *0

debug_name" QRnnNetwork/dynamic_unroll/bias/*
dtype0*
shape:Д*0
shared_name!QRnnNetwork/dynamic_unroll/bias

3QRnnNetwork/dynamic_unroll/bias/Read/ReadVariableOpReadVariableOpQRnnNetwork/dynamic_unroll/bias*
_output_shapes	
:Д*
dtype0
ё
+QRnnNetwork/dynamic_unroll/recurrent_kernelVarHandleOp*
_output_shapes
: *<

debug_name.,QRnnNetwork/dynamic_unroll/recurrent_kernel/*
dtype0*
shape:	MД*<
shared_name-+QRnnNetwork/dynamic_unroll/recurrent_kernel
Ќ
?QRnnNetwork/dynamic_unroll/recurrent_kernel/Read/ReadVariableOpReadVariableOp+QRnnNetwork/dynamic_unroll/recurrent_kernel*
_output_shapes
:	MД*
dtype0
д
!QRnnNetwork/dynamic_unroll/kernelVarHandleOp*
_output_shapes
: *2

debug_name$"QRnnNetwork/dynamic_unroll/kernel/*
dtype0*
shape:
Д*2
shared_name#!QRnnNetwork/dynamic_unroll/kernel

5QRnnNetwork/dynamic_unroll/kernel/Read/ReadVariableOpReadVariableOp!QRnnNetwork/dynamic_unroll/kernel* 
_output_shapes
:
Д*
dtype0
ф
(QRnnNetwork/EncodingNetwork/dense_1/biasVarHandleOp*
_output_shapes
: *9

debug_name+)QRnnNetwork/EncodingNetwork/dense_1/bias/*
dtype0*
shape:*9
shared_name*(QRnnNetwork/EncodingNetwork/dense_1/bias
Ђ
<QRnnNetwork/EncodingNetwork/dense_1/bias/Read/ReadVariableOpReadVariableOp(QRnnNetwork/EncodingNetwork/dense_1/bias*
_output_shapes	
:*
dtype0
я
*QRnnNetwork/EncodingNetwork/dense_1/kernelVarHandleOp*
_output_shapes
: *;

debug_name-+QRnnNetwork/EncodingNetwork/dense_1/kernel/*
dtype0*
shape:
б*;
shared_name,*QRnnNetwork/EncodingNetwork/dense_1/kernel
Ћ
>QRnnNetwork/EncodingNetwork/dense_1/kernel/Read/ReadVariableOpReadVariableOp*QRnnNetwork/EncodingNetwork/dense_1/kernel* 
_output_shapes
:
б*
dtype0
о
&QRnnNetwork/EncodingNetwork/dense/biasVarHandleOp*
_output_shapes
: *7

debug_name)'QRnnNetwork/EncodingNetwork/dense/bias/*
dtype0*
shape:б*7
shared_name(&QRnnNetwork/EncodingNetwork/dense/bias

:QRnnNetwork/EncodingNetwork/dense/bias/Read/ReadVariableOpReadVariableOp&QRnnNetwork/EncodingNetwork/dense/bias*
_output_shapes	
:б*
dtype0
ш
(QRnnNetwork/EncodingNetwork/dense/kernelVarHandleOp*
_output_shapes
: *9

debug_name+)QRnnNetwork/EncodingNetwork/dense/kernel/*
dtype0*
shape:	б*9
shared_name*(QRnnNetwork/EncodingNetwork/dense/kernel
І
<QRnnNetwork/EncodingNetwork/dense/kernel/Read/ReadVariableOpReadVariableOp(QRnnNetwork/EncodingNetwork/dense/kernel*
_output_shapes
:	б*
dtype0

VariableVarHandleOp*
_output_shapes
: *

debug_name	Variable/*
dtype0	*
shape: *
shared_name
Variable
]
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes
: *
dtype0	
l
action_0_discountPlaceholder*#
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ
w
action_0_observationPlaceholder*'
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ
j
action_0_rewardPlaceholder*#
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ
m
action_0_step_typePlaceholder*#
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ
m

action_1_0Placeholder*'
_output_shapes
:џџџџџџџџџM*
dtype0*
shape:џџџџџџџџџM
m

action_1_1Placeholder*'
_output_shapes
:џџџџџџџџџM*
dtype0*
shape:џџџџџџџџџM
К
StatefulPartitionedCallStatefulPartitionedCallaction_0_discountaction_0_observationaction_0_rewardaction_0_step_type
action_1_0
action_1_1(QRnnNetwork/EncodingNetwork/dense/kernel&QRnnNetwork/EncodingNetwork/dense/bias*QRnnNetwork/EncodingNetwork/dense_1/kernel(QRnnNetwork/EncodingNetwork/dense_1/bias!QRnnNetwork/dynamic_unroll/kernel+QRnnNetwork/dynamic_unroll/recurrent_kernelQRnnNetwork/dynamic_unroll/biasQRnnNetwork/dense_2/kernelQRnnNetwork/dense_2/biasQRnnNetwork/dense_3/kernelQRnnNetwork/dense_3/bias+QRnnNetwork/num_action_project/dense/kernel)QRnnNetwork/num_action_project/dense/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *I
_output_shapes7
5:џџџџџџџџџ:џџџџџџџџџM:џџџџџџџџџM*/
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *E
f@R>
<__inference_signature_wrapper_function_with_signature_294223
]
get_initial_state_batch_sizePlaceholder*
_output_shapes
: *
dtype0*
shape: 
д
PartitionedCallPartitionedCallget_initial_state_batch_size*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:џџџџџџџџџM:џџџџџџџџџM* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *E
f@R>
<__inference_signature_wrapper_function_with_signature_294254
ѕ
PartitionedCall_1PartitionedCall*	
Tin
 *

Tout
 *
_collective_manager_ids
 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *E
f@R>
<__inference_signature_wrapper_function_with_signature_294272
А
StatefulPartitionedCall_1StatefulPartitionedCallVariable*
Tin
2*
Tout
2	*
_collective_manager_ids
 *
_output_shapes
: *#
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *E
f@R>
<__inference_signature_wrapper_function_with_signature_294267

NoOpNoOp
Н=
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ј<
valueю<Bы< Bф<
Ъ
policy_state_spec

train_step
metadata
model_variables
_all_assets

action
distribution
get_initial_state
	get_metadata

get_train_step

signatures*
* 
GA
VARIABLE_VALUEVariable%train_step/.ATTRIBUTES/VARIABLE_VALUE*
* 
b
0
1
2
3
4
5
6
7
8
9
10
11
12*
H
_policy_state_spec
_policy_step_spec
_wrapped_policy*

trace_0
trace_1* 

trace_0* 

trace_0* 
* 
* 
K

 action
!get_initial_state
"get_train_step
#get_metadata* 
nh
VARIABLE_VALUE(QRnnNetwork/EncodingNetwork/dense/kernel,model_variables/0/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE&QRnnNetwork/EncodingNetwork/dense/bias,model_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUE*QRnnNetwork/EncodingNetwork/dense_1/kernel,model_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE(QRnnNetwork/EncodingNetwork/dense_1/bias,model_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUE!QRnnNetwork/dynamic_unroll/kernel,model_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE+QRnnNetwork/dynamic_unroll/recurrent_kernel,model_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEQRnnNetwork/dynamic_unroll/bias,model_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEQRnnNetwork/dense_2/kernel,model_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEQRnnNetwork/dense_2/bias,model_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEQRnnNetwork/dense_3/kernel,model_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEQRnnNetwork/dense_3/bias-model_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE+QRnnNetwork/num_action_project/dense/kernel-model_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUE)QRnnNetwork/num_action_project/dense/bias-model_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
* 

	state
1* 
C
$
_q_network
%_policy_state_spec
&_policy_step_spec*
* 
* 
* 
* 
* 
* 
* 
* 
н
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses
-_state_spec
._input_encoder
/_lstm_network
0_output_encoder*
* 

	%state
%1* 
b
0
1
2
3
4
5
6
7
8
9
10
11
12*
b
0
1
2
3
4
5
6
7
8
9
10
11
12*
* 

1non_trainable_variables

2layers
3metrics
4layer_regularization_losses
5layer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses*
* 
* 
* 
Ќ
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses
<_postprocessing_layers*

=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses
Ccell*

D0
E1
F2*
* 
'
.0
/1
D2
E3
F4*
* 
* 
* 
 
0
1
2
3*
 
0
1
2
3*
* 

Gnon_trainable_variables

Hlayers
Imetrics
Jlayer_regularization_losses
Klayer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses*
* 
* 

L0
M1
N2*

0
1
2*

0
1
2*
* 

Onon_trainable_variables

Players
Qmetrics
Rlayer_regularization_losses
Slayer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses*
* 
* 
у
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
X__call__
*Y&call_and_return_all_conditional_losses
Z_random_generator
[
state_size

kernel
recurrent_kernel
bias*
І
\	variables
]trainable_variables
^regularization_losses
_	keras_api
`__call__
*a&call_and_return_all_conditional_losses

kernel
bias*
І
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses

kernel
bias*
І
h	variables
itrainable_variables
jregularization_losses
k	keras_api
l__call__
*m&call_and_return_all_conditional_losses

kernel
bias*
* 

L0
M1
N2*
* 
* 
* 

n	variables
otrainable_variables
pregularization_losses
q	keras_api
r__call__
*s&call_and_return_all_conditional_losses* 
І
t	variables
utrainable_variables
vregularization_losses
w	keras_api
x__call__
*y&call_and_return_all_conditional_losses

kernel
bias*
І
z	variables
{trainable_variables
|regularization_losses
}	keras_api
~__call__
*&call_and_return_all_conditional_losses

kernel
bias*
* 

C0*
* 
* 
* 

0
1
2*

0
1
2*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
T	variables
Utrainable_variables
Vregularization_losses
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses*
* 
* 
* 
* 

0
1*

0
1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
\	variables
]trainable_variables
^regularization_losses
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses*
* 
* 

0
1*

0
1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses*
* 
* 

0
1*

0
1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
h	variables
itrainable_variables
jregularization_losses
l__call__
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
n	variables
otrainable_variables
pregularization_losses
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses* 
* 
* 

0
1*

0
1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
t	variables
utrainable_variables
vregularization_losses
x__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses*
* 
* 

0
1*

0
1*
* 

non_trainable_variables
layers
 metrics
 Ёlayer_regularization_losses
Ђlayer_metrics
z	variables
{trainable_variables
|regularization_losses
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameVariable(QRnnNetwork/EncodingNetwork/dense/kernel&QRnnNetwork/EncodingNetwork/dense/bias*QRnnNetwork/EncodingNetwork/dense_1/kernel(QRnnNetwork/EncodingNetwork/dense_1/bias!QRnnNetwork/dynamic_unroll/kernel+QRnnNetwork/dynamic_unroll/recurrent_kernelQRnnNetwork/dynamic_unroll/biasQRnnNetwork/dense_2/kernelQRnnNetwork/dense_2/biasQRnnNetwork/dense_3/kernelQRnnNetwork/dense_3/bias+QRnnNetwork/num_action_project/dense/kernel)QRnnNetwork/num_action_project/dense/biasConst*
Tin
2*
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
GPU2*0J 8 *(
f#R!
__inference__traced_save_295266

StatefulPartitionedCall_3StatefulPartitionedCallsaver_filenameVariable(QRnnNetwork/EncodingNetwork/dense/kernel&QRnnNetwork/EncodingNetwork/dense/bias*QRnnNetwork/EncodingNetwork/dense_1/kernel(QRnnNetwork/EncodingNetwork/dense_1/bias!QRnnNetwork/dynamic_unroll/kernel+QRnnNetwork/dynamic_unroll/recurrent_kernelQRnnNetwork/dynamic_unroll/biasQRnnNetwork/dense_2/kernelQRnnNetwork/dense_2/biasQRnnNetwork/dense_3/kernelQRnnNetwork/dense_3/bias+QRnnNetwork/num_action_project/dense/kernel)QRnnNetwork/num_action_project/dense/bias*
Tin
2*
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
GPU2*0J 8 *+
f&R$
"__inference__traced_restore_295317в
ћ
b
__inference_<lambda>_293279!
readvariableop_resource:	 
identity	ЂReadVariableOp^
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0	T
IdentityIdentityReadVariableOp:value:0^NoOp*
T0	*
_output_shapes
: 3
NoOpNoOp^ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2 
ReadVariableOpReadVariableOp:( $
"
_user_specified_name
resource
Е
Э
"__inference_distribution_fn_295133
	step_type

reward
discount
observation
unknown
	unknown_0S
@qrnnnetwork_encodingnetwork_dense_matmul_readvariableop_resource:	бP
Aqrnnnetwork_encodingnetwork_dense_biasadd_readvariableop_resource:	бV
Bqrnnnetwork_encodingnetwork_dense_1_matmul_readvariableop_resource:
бR
Cqrnnnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resource:	W
Cqrnnnetwork_dynamic_unroll_lstm_cell_matmul_readvariableop_resource:
ДX
Eqrnnnetwork_dynamic_unroll_lstm_cell_matmul_1_readvariableop_resource:	MДS
Dqrnnnetwork_dynamic_unroll_lstm_cell_biasadd_readvariableop_resource:	ДH
5qrnnnetwork_dense_2_tensordot_readvariableop_resource:	MиB
3qrnnnetwork_dense_2_biasadd_readvariableop_resource:	иH
5qrnnnetwork_dense_3_tensordot_readvariableop_resource:	иdA
3qrnnnetwork_dense_3_biasadd_readvariableop_resource:dX
Fqrnnnetwork_num_action_project_dense_tensordot_readvariableop_resource:dR
Dqrnnnetwork_num_action_project_dense_biasadd_readvariableop_resource:
identity

identity_1

identity_2

identity_3

identity_4Ђ8QRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpЂ7QRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpЂ:QRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpЂ9QRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpЂ*QRnnNetwork/dense_2/BiasAdd/ReadVariableOpЂ,QRnnNetwork/dense_2/Tensordot/ReadVariableOpЂ*QRnnNetwork/dense_3/BiasAdd/ReadVariableOpЂ,QRnnNetwork/dense_3/Tensordot/ReadVariableOpЂ;QRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOpЂ:QRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOpЂ<QRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOpЂ;QRnnNetwork/num_action_project/dense/BiasAdd/ReadVariableOpЂ=QRnnNetwork/num_action_project/dense/Tensordot/ReadVariableOpK
ShapeShapediscount*
T0*
_output_shapes
::эЯ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskT
packedPackstrided_slice:output:0*
N*
T0*
_output_shapes
:Y
shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:MM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
concatConcatV2packed:output:0shape_as_tensor:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    f
zerosFillconcat:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџM[
shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB:MO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
concat_1ConcatV2packed:output:0shape_as_tensor_1:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zeros_1Fillconcat_1:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџMI
Equal/yConst*
_output_shapes
: *
dtype0*
value	B : Y
EqualEqual	step_typeEqual/y:output:0*
T0*#
_output_shapes
:џџџџџџџџџF
RankConst*
_output_shapes
: *
dtype0*
value	B :О
PartitionedCallPartitionedCallzeros:output:0unknownRank:output:0	Equal:z:0*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџM* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference_per_field_where_294909Ф
PartitionedCall_1PartitionedCallzeros_1:output:0	unknown_0Rank:output:0	Equal:z:0*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџM* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference_per_field_where_294909M
Shape_1Shapediscount*
T0*
_output_shapes
::эЯ_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
packed_1Packstrided_slice_1:output:0*
N*
T0*
_output_shapes
:[
shape_as_tensor_2Const*
_output_shapes
:*
dtype0*
valueB:MO
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
concat_2ConcatV2packed_1:output:0shape_as_tensor_2:output:0concat_2/axis:output:0*
N*
T0*
_output_shapes
:R
zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zeros_2Fillconcat_2:output:0zeros_2/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџM[
shape_as_tensor_3Const*
_output_shapes
:*
dtype0*
valueB:MO
concat_3/axisConst*
_output_shapes
: *
dtype0*
value	B : 
concat_3ConcatV2packed_1:output:0shape_as_tensor_3:output:0concat_3/axis:output:0*
N*
T0*
_output_shapes
:R
zeros_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zeros_3Fillconcat_3:output:0zeros_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџMK
	Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : ]
Equal_1Equal	step_typeEqual_1/y:output:0*
T0*#
_output_shapes
:џџџџџџџџџH
Rank_1Const*
_output_shapes
: *
dtype0*
value	B :з
PartitionedCall_2PartitionedCallzeros_2:output:0PartitionedCall:output:0Rank_1:output:0Equal_1:z:0*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџM* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference_per_field_where_294949й
PartitionedCall_3PartitionedCallzeros_3:output:0PartitionedCall_1:output:0Rank_1:output:0Equal_1:z:0*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџM* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference_per_field_where_294949\
QRnnNetwork/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :
QRnnNetwork/ExpandDims
ExpandDimsobservation#QRnnNetwork/ExpandDims/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџ^
QRnnNetwork/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :
QRnnNetwork/ExpandDims_1
ExpandDims	step_type%QRnnNetwork/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
/QRnnNetwork/EncodingNetwork/batch_flatten/ShapeShapeQRnnNetwork/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	:эа
7QRnnNetwork/EncodingNetwork/batch_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   б
1QRnnNetwork/EncodingNetwork/batch_flatten/ReshapeReshapeQRnnNetwork/ExpandDims:output:0@QRnnNetwork/EncodingNetwork/batch_flatten/Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџz
)QRnnNetwork/EncodingNetwork/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   и
+QRnnNetwork/EncodingNetwork/flatten/ReshapeReshape:QRnnNetwork/EncodingNetwork/batch_flatten/Reshape:output:02QRnnNetwork/EncodingNetwork/flatten/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџЙ
7QRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpReadVariableOp@qrnnnetwork_encodingnetwork_dense_matmul_readvariableop_resource*
_output_shapes
:	б*
dtype0м
(QRnnNetwork/EncodingNetwork/dense/MatMulMatMul4QRnnNetwork/EncodingNetwork/flatten/Reshape:output:0?QRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџбЗ
8QRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpReadVariableOpAqrnnnetwork_encodingnetwork_dense_biasadd_readvariableop_resource*
_output_shapes	
:б*
dtype0н
)QRnnNetwork/EncodingNetwork/dense/BiasAddBiasAdd2QRnnNetwork/EncodingNetwork/dense/MatMul:product:0@QRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџб
&QRnnNetwork/EncodingNetwork/dense/SeluSelu2QRnnNetwork/EncodingNetwork/dense/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџбО
9QRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpReadVariableOpBqrnnnetwork_encodingnetwork_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
б*
dtype0р
*QRnnNetwork/EncodingNetwork/dense_1/MatMulMatMul4QRnnNetwork/EncodingNetwork/dense/Selu:activations:0AQRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЛ
:QRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpReadVariableOpCqrnnnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0у
+QRnnNetwork/EncodingNetwork/dense_1/BiasAddBiasAdd4QRnnNetwork/EncodingNetwork/dense_1/MatMul:product:0BQRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
(QRnnNetwork/EncodingNetwork/dense_1/SeluSelu4QRnnNetwork/EncodingNetwork/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
?QRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
AQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
AQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Љ
9QRnnNetwork/EncodingNetwork/batch_unflatten/strided_sliceStridedSlice8QRnnNetwork/EncodingNetwork/batch_flatten/Shape:output:0HQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack:output:0JQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_1:output:0JQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_maskЕ
1QRnnNetwork/EncodingNetwork/batch_unflatten/ShapeShape6QRnnNetwork/EncodingNetwork/dense_1/Selu:activations:0*
T0*
_output_shapes
:*
out_type0	:эа
AQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
CQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
CQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Б
;QRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1StridedSlice:QRnnNetwork/EncodingNetwork/batch_unflatten/Shape:output:0JQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack:output:0LQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_1:output:0LQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
end_masky
7QRnnNetwork/EncodingNetwork/batch_unflatten/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : И
2QRnnNetwork/EncodingNetwork/batch_unflatten/concatConcatV2BQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice:output:0DQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1:output:0@QRnnNetwork/EncodingNetwork/batch_unflatten/concat/axis:output:0*
N*
T0	*
_output_shapes
:ј
3QRnnNetwork/EncodingNetwork/batch_unflatten/ReshapeReshape6QRnnNetwork/EncodingNetwork/dense_1/Selu:activations:0;QRnnNetwork/EncodingNetwork/batch_unflatten/concat:output:0*
Tshape0	*
T0*,
_output_shapes
:џџџџџџџџџT
QRnnNetwork/mask/yConst*
_output_shapes
: *
dtype0*
value	B : 
QRnnNetwork/maskEqual!QRnnNetwork/ExpandDims_1:output:0QRnnNetwork/mask/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџa
QRnnNetwork/dynamic_unroll/RankConst*
_output_shapes
: *
dtype0*
value	B :h
&QRnnNetwork/dynamic_unroll/range/startConst*
_output_shapes
: *
dtype0*
value	B :h
&QRnnNetwork/dynamic_unroll/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :б
 QRnnNetwork/dynamic_unroll/rangeRange/QRnnNetwork/dynamic_unroll/range/start:output:0(QRnnNetwork/dynamic_unroll/Rank:output:0/QRnnNetwork/dynamic_unroll/range/delta:output:0*
_output_shapes
:{
*QRnnNetwork/dynamic_unroll/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB"       h
&QRnnNetwork/dynamic_unroll/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ь
!QRnnNetwork/dynamic_unroll/concatConcatV23QRnnNetwork/dynamic_unroll/concat/values_0:output:0)QRnnNetwork/dynamic_unroll/range:output:0/QRnnNetwork/dynamic_unroll/concat/axis:output:0*
N*
T0*
_output_shapes
:в
$QRnnNetwork/dynamic_unroll/transpose	Transpose<QRnnNetwork/EncodingNetwork/batch_unflatten/Reshape:output:0*QRnnNetwork/dynamic_unroll/concat:output:0*
T0*,
_output_shapes
:џџџџџџџџџ
 QRnnNetwork/dynamic_unroll/ShapeShape(QRnnNetwork/dynamic_unroll/transpose:y:0*
T0*
_output_shapes
::эЯx
.QRnnNetwork/dynamic_unroll/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:z
0QRnnNetwork/dynamic_unroll/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0QRnnNetwork/dynamic_unroll/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:и
(QRnnNetwork/dynamic_unroll/strided_sliceStridedSlice)QRnnNetwork/dynamic_unroll/Shape:output:07QRnnNetwork/dynamic_unroll/strided_slice/stack:output:09QRnnNetwork/dynamic_unroll/strided_slice/stack_1:output:09QRnnNetwork/dynamic_unroll/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask|
+QRnnNetwork/dynamic_unroll/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       Б
&QRnnNetwork/dynamic_unroll/transpose_1	TransposeQRnnNetwork/mask:z:04QRnnNetwork/dynamic_unroll/transpose_1/perm:output:0*
T0
*'
_output_shapes
:џџџџџџџџџk
)QRnnNetwork/dynamic_unroll/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :MФ
'QRnnNetwork/dynamic_unroll/zeros/packedPack1QRnnNetwork/dynamic_unroll/strided_slice:output:02QRnnNetwork/dynamic_unroll/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:k
&QRnnNetwork/dynamic_unroll/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Н
 QRnnNetwork/dynamic_unroll/zerosFill0QRnnNetwork/dynamic_unroll/zeros/packed:output:0/QRnnNetwork/dynamic_unroll/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџMm
+QRnnNetwork/dynamic_unroll/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :MШ
)QRnnNetwork/dynamic_unroll/zeros_1/packedPack1QRnnNetwork/dynamic_unroll/strided_slice:output:04QRnnNetwork/dynamic_unroll/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:m
(QRnnNetwork/dynamic_unroll/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    У
"QRnnNetwork/dynamic_unroll/zeros_1Fill2QRnnNetwork/dynamic_unroll/zeros_1/packed:output:01QRnnNetwork/dynamic_unroll/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџMЁ
"QRnnNetwork/dynamic_unroll/SqueezeSqueeze(QRnnNetwork/dynamic_unroll/transpose:y:0*
T0*(
_output_shapes
:џџџџџџџџџ*
squeeze_dims
  
$QRnnNetwork/dynamic_unroll/Squeeze_1Squeeze*QRnnNetwork/dynamic_unroll/transpose_1:y:0*
T0
*#
_output_shapes
:џџџџџџџџџ*
squeeze_dims
 г
!QRnnNetwork/dynamic_unroll/SelectSelect-QRnnNetwork/dynamic_unroll/Squeeze_1:output:0)QRnnNetwork/dynamic_unroll/zeros:output:0PartitionedCall_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџMз
#QRnnNetwork/dynamic_unroll/Select_1Select-QRnnNetwork/dynamic_unroll/Squeeze_1:output:0+QRnnNetwork/dynamic_unroll/zeros_1:output:0PartitionedCall_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџMР
:QRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOpReadVariableOpCqrnnnetwork_dynamic_unroll_lstm_cell_matmul_readvariableop_resource* 
_output_shapes
:
Д*
dtype0й
+QRnnNetwork/dynamic_unroll/lstm_cell/MatMulMatMul+QRnnNetwork/dynamic_unroll/Squeeze:output:0BQRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџДУ
<QRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpEqrnnnetwork_dynamic_unroll_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes
:	MД*
dtype0м
-QRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1MatMul*QRnnNetwork/dynamic_unroll/Select:output:0DQRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџДд
(QRnnNetwork/dynamic_unroll/lstm_cell/addAddV25QRnnNetwork/dynamic_unroll/lstm_cell/MatMul:product:07QRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџДН
;QRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOpReadVariableOpDqrnnnetwork_dynamic_unroll_lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:Д*
dtype0н
,QRnnNetwork/dynamic_unroll/lstm_cell/BiasAddBiasAdd,QRnnNetwork/dynamic_unroll/lstm_cell/add:z:0CQRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџДv
4QRnnNetwork/dynamic_unroll/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ѕ
*QRnnNetwork/dynamic_unroll/lstm_cell/splitSplit=QRnnNetwork/dynamic_unroll/lstm_cell/split/split_dim:output:05QRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџM:џџџџџџџџџM:џџџџџџџџџM:џџџџџџџџџM*
	num_split
,QRnnNetwork/dynamic_unroll/lstm_cell/SigmoidSigmoid3QRnnNetwork/dynamic_unroll/lstm_cell/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџM 
.QRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_1Sigmoid3QRnnNetwork/dynamic_unroll/lstm_cell/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџMУ
(QRnnNetwork/dynamic_unroll/lstm_cell/mulMul2QRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_1:y:0,QRnnNetwork/dynamic_unroll/Select_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџM
)QRnnNetwork/dynamic_unroll/lstm_cell/TanhTanh3QRnnNetwork/dynamic_unroll/lstm_cell/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџMФ
*QRnnNetwork/dynamic_unroll/lstm_cell/mul_1Mul0QRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid:y:0-QRnnNetwork/dynamic_unroll/lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџMУ
*QRnnNetwork/dynamic_unroll/lstm_cell/add_1AddV2,QRnnNetwork/dynamic_unroll/lstm_cell/mul:z:0.QRnnNetwork/dynamic_unroll/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџM 
.QRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_2Sigmoid3QRnnNetwork/dynamic_unroll/lstm_cell/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџM
+QRnnNetwork/dynamic_unroll/lstm_cell/Tanh_1Tanh.QRnnNetwork/dynamic_unroll/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџMШ
*QRnnNetwork/dynamic_unroll/lstm_cell/mul_2Mul2QRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_2:y:0/QRnnNetwork/dynamic_unroll/lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџMk
)QRnnNetwork/dynamic_unroll/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :Э
%QRnnNetwork/dynamic_unroll/ExpandDims
ExpandDims.QRnnNetwork/dynamic_unroll/lstm_cell/mul_2:z:02QRnnNetwork/dynamic_unroll/ExpandDims/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџMЃ
,QRnnNetwork/dense_2/Tensordot/ReadVariableOpReadVariableOp5qrnnnetwork_dense_2_tensordot_readvariableop_resource*
_output_shapes
:	Mи*
dtype0l
"QRnnNetwork/dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:s
"QRnnNetwork/dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       
#QRnnNetwork/dense_2/Tensordot/ShapeShape.QRnnNetwork/dynamic_unroll/ExpandDims:output:0*
T0*
_output_shapes
::эЯm
+QRnnNetwork/dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
&QRnnNetwork/dense_2/Tensordot/GatherV2GatherV2,QRnnNetwork/dense_2/Tensordot/Shape:output:0+QRnnNetwork/dense_2/Tensordot/free:output:04QRnnNetwork/dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:o
-QRnnNetwork/dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
(QRnnNetwork/dense_2/Tensordot/GatherV2_1GatherV2,QRnnNetwork/dense_2/Tensordot/Shape:output:0+QRnnNetwork/dense_2/Tensordot/axes:output:06QRnnNetwork/dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:m
#QRnnNetwork/dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: Њ
"QRnnNetwork/dense_2/Tensordot/ProdProd/QRnnNetwork/dense_2/Tensordot/GatherV2:output:0,QRnnNetwork/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: o
%QRnnNetwork/dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: А
$QRnnNetwork/dense_2/Tensordot/Prod_1Prod1QRnnNetwork/dense_2/Tensordot/GatherV2_1:output:0.QRnnNetwork/dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: k
)QRnnNetwork/dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ь
$QRnnNetwork/dense_2/Tensordot/concatConcatV2+QRnnNetwork/dense_2/Tensordot/free:output:0+QRnnNetwork/dense_2/Tensordot/axes:output:02QRnnNetwork/dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:Е
#QRnnNetwork/dense_2/Tensordot/stackPack+QRnnNetwork/dense_2/Tensordot/Prod:output:0-QRnnNetwork/dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Щ
'QRnnNetwork/dense_2/Tensordot/transpose	Transpose.QRnnNetwork/dynamic_unroll/ExpandDims:output:0-QRnnNetwork/dense_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџMЦ
%QRnnNetwork/dense_2/Tensordot/ReshapeReshape+QRnnNetwork/dense_2/Tensordot/transpose:y:0,QRnnNetwork/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџЧ
$QRnnNetwork/dense_2/Tensordot/MatMulMatMul.QRnnNetwork/dense_2/Tensordot/Reshape:output:04QRnnNetwork/dense_2/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџиp
%QRnnNetwork/dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:иm
+QRnnNetwork/dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ї
&QRnnNetwork/dense_2/Tensordot/concat_1ConcatV2/QRnnNetwork/dense_2/Tensordot/GatherV2:output:0.QRnnNetwork/dense_2/Tensordot/Const_2:output:04QRnnNetwork/dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Р
QRnnNetwork/dense_2/TensordotReshape.QRnnNetwork/dense_2/Tensordot/MatMul:product:0/QRnnNetwork/dense_2/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:џџџџџџџџџи
*QRnnNetwork/dense_2/BiasAdd/ReadVariableOpReadVariableOp3qrnnnetwork_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:и*
dtype0Й
QRnnNetwork/dense_2/BiasAddBiasAdd&QRnnNetwork/dense_2/Tensordot:output:02QRnnNetwork/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџи}
QRnnNetwork/dense_2/SeluSelu$QRnnNetwork/dense_2/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџиЃ
,QRnnNetwork/dense_3/Tensordot/ReadVariableOpReadVariableOp5qrnnnetwork_dense_3_tensordot_readvariableop_resource*
_output_shapes
:	иd*
dtype0l
"QRnnNetwork/dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:s
"QRnnNetwork/dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       
#QRnnNetwork/dense_3/Tensordot/ShapeShape&QRnnNetwork/dense_2/Selu:activations:0*
T0*
_output_shapes
::эЯm
+QRnnNetwork/dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
&QRnnNetwork/dense_3/Tensordot/GatherV2GatherV2,QRnnNetwork/dense_3/Tensordot/Shape:output:0+QRnnNetwork/dense_3/Tensordot/free:output:04QRnnNetwork/dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:o
-QRnnNetwork/dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
(QRnnNetwork/dense_3/Tensordot/GatherV2_1GatherV2,QRnnNetwork/dense_3/Tensordot/Shape:output:0+QRnnNetwork/dense_3/Tensordot/axes:output:06QRnnNetwork/dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:m
#QRnnNetwork/dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: Њ
"QRnnNetwork/dense_3/Tensordot/ProdProd/QRnnNetwork/dense_3/Tensordot/GatherV2:output:0,QRnnNetwork/dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: o
%QRnnNetwork/dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: А
$QRnnNetwork/dense_3/Tensordot/Prod_1Prod1QRnnNetwork/dense_3/Tensordot/GatherV2_1:output:0.QRnnNetwork/dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: k
)QRnnNetwork/dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ь
$QRnnNetwork/dense_3/Tensordot/concatConcatV2+QRnnNetwork/dense_3/Tensordot/free:output:0+QRnnNetwork/dense_3/Tensordot/axes:output:02QRnnNetwork/dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:Е
#QRnnNetwork/dense_3/Tensordot/stackPack+QRnnNetwork/dense_3/Tensordot/Prod:output:0-QRnnNetwork/dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Т
'QRnnNetwork/dense_3/Tensordot/transpose	Transpose&QRnnNetwork/dense_2/Selu:activations:0-QRnnNetwork/dense_3/Tensordot/concat:output:0*
T0*,
_output_shapes
:џџџџџџџџџиЦ
%QRnnNetwork/dense_3/Tensordot/ReshapeReshape+QRnnNetwork/dense_3/Tensordot/transpose:y:0,QRnnNetwork/dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџЦ
$QRnnNetwork/dense_3/Tensordot/MatMulMatMul.QRnnNetwork/dense_3/Tensordot/Reshape:output:04QRnnNetwork/dense_3/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџdo
%QRnnNetwork/dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:dm
+QRnnNetwork/dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ї
&QRnnNetwork/dense_3/Tensordot/concat_1ConcatV2/QRnnNetwork/dense_3/Tensordot/GatherV2:output:0.QRnnNetwork/dense_3/Tensordot/Const_2:output:04QRnnNetwork/dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:П
QRnnNetwork/dense_3/TensordotReshape.QRnnNetwork/dense_3/Tensordot/MatMul:product:0/QRnnNetwork/dense_3/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџd
*QRnnNetwork/dense_3/BiasAdd/ReadVariableOpReadVariableOp3qrnnnetwork_dense_3_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0И
QRnnNetwork/dense_3/BiasAddBiasAdd&QRnnNetwork/dense_3/Tensordot:output:02QRnnNetwork/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџd|
QRnnNetwork/dense_3/SeluSelu$QRnnNetwork/dense_3/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџdФ
=QRnnNetwork/num_action_project/dense/Tensordot/ReadVariableOpReadVariableOpFqrnnnetwork_num_action_project_dense_tensordot_readvariableop_resource*
_output_shapes

:d*
dtype0}
3QRnnNetwork/num_action_project/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
3QRnnNetwork/num_action_project/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       
4QRnnNetwork/num_action_project/dense/Tensordot/ShapeShape&QRnnNetwork/dense_3/Selu:activations:0*
T0*
_output_shapes
::эЯ~
<QRnnNetwork/num_action_project/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Я
7QRnnNetwork/num_action_project/dense/Tensordot/GatherV2GatherV2=QRnnNetwork/num_action_project/dense/Tensordot/Shape:output:0<QRnnNetwork/num_action_project/dense/Tensordot/free:output:0EQRnnNetwork/num_action_project/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
>QRnnNetwork/num_action_project/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : г
9QRnnNetwork/num_action_project/dense/Tensordot/GatherV2_1GatherV2=QRnnNetwork/num_action_project/dense/Tensordot/Shape:output:0<QRnnNetwork/num_action_project/dense/Tensordot/axes:output:0GQRnnNetwork/num_action_project/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:~
4QRnnNetwork/num_action_project/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: н
3QRnnNetwork/num_action_project/dense/Tensordot/ProdProd@QRnnNetwork/num_action_project/dense/Tensordot/GatherV2:output:0=QRnnNetwork/num_action_project/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 
6QRnnNetwork/num_action_project/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: у
5QRnnNetwork/num_action_project/dense/Tensordot/Prod_1ProdBQRnnNetwork/num_action_project/dense/Tensordot/GatherV2_1:output:0?QRnnNetwork/num_action_project/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: |
:QRnnNetwork/num_action_project/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : А
5QRnnNetwork/num_action_project/dense/Tensordot/concatConcatV2<QRnnNetwork/num_action_project/dense/Tensordot/free:output:0<QRnnNetwork/num_action_project/dense/Tensordot/axes:output:0CQRnnNetwork/num_action_project/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:ш
4QRnnNetwork/num_action_project/dense/Tensordot/stackPack<QRnnNetwork/num_action_project/dense/Tensordot/Prod:output:0>QRnnNetwork/num_action_project/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:у
8QRnnNetwork/num_action_project/dense/Tensordot/transpose	Transpose&QRnnNetwork/dense_3/Selu:activations:0>QRnnNetwork/num_action_project/dense/Tensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџdљ
6QRnnNetwork/num_action_project/dense/Tensordot/ReshapeReshape<QRnnNetwork/num_action_project/dense/Tensordot/transpose:y:0=QRnnNetwork/num_action_project/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџљ
5QRnnNetwork/num_action_project/dense/Tensordot/MatMulMatMul?QRnnNetwork/num_action_project/dense/Tensordot/Reshape:output:0EQRnnNetwork/num_action_project/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
6QRnnNetwork/num_action_project/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:~
<QRnnNetwork/num_action_project/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Л
7QRnnNetwork/num_action_project/dense/Tensordot/concat_1ConcatV2@QRnnNetwork/num_action_project/dense/Tensordot/GatherV2:output:0?QRnnNetwork/num_action_project/dense/Tensordot/Const_2:output:0EQRnnNetwork/num_action_project/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:ђ
.QRnnNetwork/num_action_project/dense/TensordotReshape?QRnnNetwork/num_action_project/dense/Tensordot/MatMul:product:0@QRnnNetwork/num_action_project/dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџМ
;QRnnNetwork/num_action_project/dense/BiasAdd/ReadVariableOpReadVariableOpDqrnnnetwork_num_action_project_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ы
,QRnnNetwork/num_action_project/dense/BiasAddBiasAdd7QRnnNetwork/num_action_project/dense/Tensordot:output:0CQRnnNetwork/num_action_project/dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ
QRnnNetwork/SqueezeSqueeze5QRnnNetwork/num_action_project/dense/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
squeeze_dims
l
!Categorical/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
Categorical/mode/ArgMaxArgMaxQRnnNetwork/Squeeze:output:0*Categorical/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:џџџџџџџџџ|
Categorical/mode/CastCast Categorical/mode/ArgMax:output:0*

DstT0*

SrcT0	*#
_output_shapes
:џџџџџџџџџW
Deterministic/atolConst*
_output_shapes
: *
dtype0*
valueB
 *    W
Deterministic/rtolConst*
_output_shapes
: *
dtype0*
valueB
 *    Y
IdentityIdentityDeterministic/atol:output:0^NoOp*
T0*
_output_shapes
: f

Identity_1IdentityCategorical/mode/Cast:y:0^NoOp*
T0*#
_output_shapes
:џџџџџџџџџ[

Identity_2IdentityDeterministic/rtol:output:0^NoOp*
T0*
_output_shapes
: 

Identity_3Identity.QRnnNetwork/dynamic_unroll/lstm_cell/mul_2:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџM

Identity_4Identity.QRnnNetwork/dynamic_unroll/lstm_cell/add_1:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџM
NoOpNoOp9^QRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp8^QRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp;^QRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp:^QRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp+^QRnnNetwork/dense_2/BiasAdd/ReadVariableOp-^QRnnNetwork/dense_2/Tensordot/ReadVariableOp+^QRnnNetwork/dense_3/BiasAdd/ReadVariableOp-^QRnnNetwork/dense_3/Tensordot/ReadVariableOp<^QRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOp;^QRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOp=^QRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOp<^QRnnNetwork/num_action_project/dense/BiasAdd/ReadVariableOp>^QRnnNetwork/num_action_project/dense/Tensordot/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџM:џџџџџџџџџM: : : : : : : : : : : : : 2t
8QRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp8QRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp2r
7QRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp7QRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp2x
:QRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp:QRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp2v
9QRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp9QRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp2X
*QRnnNetwork/dense_2/BiasAdd/ReadVariableOp*QRnnNetwork/dense_2/BiasAdd/ReadVariableOp2\
,QRnnNetwork/dense_2/Tensordot/ReadVariableOp,QRnnNetwork/dense_2/Tensordot/ReadVariableOp2X
*QRnnNetwork/dense_3/BiasAdd/ReadVariableOp*QRnnNetwork/dense_3/BiasAdd/ReadVariableOp2\
,QRnnNetwork/dense_3/Tensordot/ReadVariableOp,QRnnNetwork/dense_3/Tensordot/ReadVariableOp2z
;QRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOp;QRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOp2x
:QRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOp:QRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOp2|
<QRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOp<QRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOp2z
;QRnnNetwork/num_action_project/dense/BiasAdd/ReadVariableOp;QRnnNetwork/num_action_project/dense/BiasAdd/ReadVariableOp2~
=QRnnNetwork/num_action_project/dense/Tensordot/ReadVariableOp=QRnnNetwork/num_action_project/dense/Tensordot/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:JF
'
_output_shapes
:џџџџџџџџџM

_user_specified_name1:JF
'
_output_shapes
:џџџџџџџџџM

_user_specified_name0:TP
'
_output_shapes
:џџџџџџџџџ
%
_user_specified_nameobservation:MI
#
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
discount:KG
#
_output_shapes
:џџџџџџџџџ
 
_user_specified_namereward:N J
#
_output_shapes
:џџџџџџџџџ
#
_user_specified_name	step_type
\

__inference_<lambda>_293281*(
_construction_contextkEagerRuntime*
_input_shapes 
Ъ
_
"__inference_per_field_where_293898
t
f
sub_rank
shape_equal

identityY
assert_rank_at_least/ShapeShapet*
T0*
_output_shapes
::эЯa
Cassert_rank_at_least/assert_type/statically_determined_correct_typeNoOp*
_output_shapes
 R
4assert_rank_at_least/static_checks_determined_all_okNoOp*
_output_shapes
 F
RankConst*
_output_shapes
: *
dtype0*
value	B :D
subSubRank:output:0sub_rank*
T0*
_output_shapes
: N
ShapeShapeshape_equal*
T0
*
_output_shapes
::эЯe
ones/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџb
ones/ReshapeReshapesub:z:0ones/Reshape/shape:output:0*
T0*
_output_shapes
:L

ones/ConstConst*
_output_shapes
: *
dtype0*
value	B :f
onesFillones/Reshape:output:0ones/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ~
concatConcatV2Shape:output:0ones:output:0concat/axis:output:0*
N*
T0*#
_output_shapes
:џџџџџџџџџS
ReshapeReshapeshape_equalconcat:output:0*
T0
*
_output_shapes
:O
SelectV2SelectV2Reshape:output:0tf*
T0*
_output_shapes
:J
IdentityIdentitySelectV2:output:0*
T0*
_output_shapes
:"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:џџџџџџџџџM:џџџџџџџџџM: :џџџџџџџџџ:JF
#
_output_shapes
:џџџџџџџџџ

_user_specified_nameEqual:<8

_output_shapes
: 

_user_specified_nameRank:JF
'
_output_shapes
:џџџџџџџџџM

_user_specified_namef:J F
'
_output_shapes
:џџџџџџџџџM

_user_specified_namet
Ў
l
<__inference_signature_wrapper_function_with_signature_294254

batch_size
identity

identity_1А
PartitionedCallPartitionedCall
batch_size*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:џџџџџџџџџM:џџџџџџџџџM* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *3
f.R,
*__inference_function_with_signature_294246`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџMb

Identity_1IdentityPartitionedCall:output:1*
T0*'
_output_shapes
:џџџџџџџџџM"!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
і
|
<__inference_signature_wrapper_function_with_signature_294267
unknown:	 
identity	ЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallunknown*
Tin
2*
Tout
2	*
_collective_manager_ids
 *
_output_shapes
: *#
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *3
f.R,
*__inference_function_with_signature_294260^
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0	*
_output_shapes
: <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_name294263
Їг
Ж
__inference_action_294149
	time_step
time_step_1
time_step_2
time_step_3
policy_state
policy_state_1S
@qrnnnetwork_encodingnetwork_dense_matmul_readvariableop_resource:	бP
Aqrnnnetwork_encodingnetwork_dense_biasadd_readvariableop_resource:	бV
Bqrnnnetwork_encodingnetwork_dense_1_matmul_readvariableop_resource:
бR
Cqrnnnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resource:	W
Cqrnnnetwork_dynamic_unroll_lstm_cell_matmul_readvariableop_resource:
ДX
Eqrnnnetwork_dynamic_unroll_lstm_cell_matmul_1_readvariableop_resource:	MДS
Dqrnnnetwork_dynamic_unroll_lstm_cell_biasadd_readvariableop_resource:	ДH
5qrnnnetwork_dense_2_tensordot_readvariableop_resource:	MиB
3qrnnnetwork_dense_2_biasadd_readvariableop_resource:	иH
5qrnnnetwork_dense_3_tensordot_readvariableop_resource:	иdA
3qrnnnetwork_dense_3_biasadd_readvariableop_resource:dX
Fqrnnnetwork_num_action_project_dense_tensordot_readvariableop_resource:dR
Dqrnnnetwork_num_action_project_dense_biasadd_readvariableop_resource:
identity

identity_1

identity_2Ђ8QRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpЂ7QRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpЂ:QRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpЂ9QRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpЂ*QRnnNetwork/dense_2/BiasAdd/ReadVariableOpЂ,QRnnNetwork/dense_2/Tensordot/ReadVariableOpЂ*QRnnNetwork/dense_3/BiasAdd/ReadVariableOpЂ,QRnnNetwork/dense_3/Tensordot/ReadVariableOpЂ;QRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOpЂ:QRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOpЂ<QRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOpЂ;QRnnNetwork/num_action_project/dense/BiasAdd/ReadVariableOpЂ=QRnnNetwork/num_action_project/dense/Tensordot/ReadVariableOpN
ShapeShapetime_step_2*
T0*
_output_shapes
::эЯ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskT
packedPackstrided_slice:output:0*
N*
T0*
_output_shapes
:Y
shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:MM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
concatConcatV2packed:output:0shape_as_tensor:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    f
zerosFillconcat:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџM[
shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB:MO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
concat_1ConcatV2packed:output:0shape_as_tensor_1:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zeros_1Fillconcat_1:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџMI
Equal/yConst*
_output_shapes
: *
dtype0*
value	B : Y
EqualEqual	time_stepEqual/y:output:0*
T0*#
_output_shapes
:џџџџџџџџџF
RankConst*
_output_shapes
: *
dtype0*
value	B :У
PartitionedCallPartitionedCallzeros:output:0policy_stateRank:output:0	Equal:z:0*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџM* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference_per_field_where_293898Щ
PartitionedCall_1PartitionedCallzeros_1:output:0policy_state_1Rank:output:0	Equal:z:0*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџM* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference_per_field_where_293898P
Shape_1Shapetime_step_2*
T0*
_output_shapes
::эЯ_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
packed_1Packstrided_slice_1:output:0*
N*
T0*
_output_shapes
:[
shape_as_tensor_2Const*
_output_shapes
:*
dtype0*
valueB:MO
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
concat_2ConcatV2packed_1:output:0shape_as_tensor_2:output:0concat_2/axis:output:0*
N*
T0*
_output_shapes
:R
zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zeros_2Fillconcat_2:output:0zeros_2/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџM[
shape_as_tensor_3Const*
_output_shapes
:*
dtype0*
valueB:MO
concat_3/axisConst*
_output_shapes
: *
dtype0*
value	B : 
concat_3ConcatV2packed_1:output:0shape_as_tensor_3:output:0concat_3/axis:output:0*
N*
T0*
_output_shapes
:R
zeros_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zeros_3Fillconcat_3:output:0zeros_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџMK
	Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : ]
Equal_1Equal	time_stepEqual_1/y:output:0*
T0*#
_output_shapes
:џџџџџџџџџH
Rank_1Const*
_output_shapes
: *
dtype0*
value	B :з
PartitionedCall_2PartitionedCallzeros_2:output:0PartitionedCall:output:0Rank_1:output:0Equal_1:z:0*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџM* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference_per_field_where_293938й
PartitionedCall_3PartitionedCallzeros_3:output:0PartitionedCall_1:output:0Rank_1:output:0Equal_1:z:0*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџM* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference_per_field_where_293938\
QRnnNetwork/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :
QRnnNetwork/ExpandDims
ExpandDimstime_step_3#QRnnNetwork/ExpandDims/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџ^
QRnnNetwork/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :
QRnnNetwork/ExpandDims_1
ExpandDims	time_step%QRnnNetwork/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
/QRnnNetwork/EncodingNetwork/batch_flatten/ShapeShapeQRnnNetwork/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	:эа
7QRnnNetwork/EncodingNetwork/batch_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   б
1QRnnNetwork/EncodingNetwork/batch_flatten/ReshapeReshapeQRnnNetwork/ExpandDims:output:0@QRnnNetwork/EncodingNetwork/batch_flatten/Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџz
)QRnnNetwork/EncodingNetwork/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   и
+QRnnNetwork/EncodingNetwork/flatten/ReshapeReshape:QRnnNetwork/EncodingNetwork/batch_flatten/Reshape:output:02QRnnNetwork/EncodingNetwork/flatten/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџЙ
7QRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpReadVariableOp@qrnnnetwork_encodingnetwork_dense_matmul_readvariableop_resource*
_output_shapes
:	б*
dtype0м
(QRnnNetwork/EncodingNetwork/dense/MatMulMatMul4QRnnNetwork/EncodingNetwork/flatten/Reshape:output:0?QRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџбЗ
8QRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpReadVariableOpAqrnnnetwork_encodingnetwork_dense_biasadd_readvariableop_resource*
_output_shapes	
:б*
dtype0н
)QRnnNetwork/EncodingNetwork/dense/BiasAddBiasAdd2QRnnNetwork/EncodingNetwork/dense/MatMul:product:0@QRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџб
&QRnnNetwork/EncodingNetwork/dense/SeluSelu2QRnnNetwork/EncodingNetwork/dense/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџбО
9QRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpReadVariableOpBqrnnnetwork_encodingnetwork_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
б*
dtype0р
*QRnnNetwork/EncodingNetwork/dense_1/MatMulMatMul4QRnnNetwork/EncodingNetwork/dense/Selu:activations:0AQRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЛ
:QRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpReadVariableOpCqrnnnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0у
+QRnnNetwork/EncodingNetwork/dense_1/BiasAddBiasAdd4QRnnNetwork/EncodingNetwork/dense_1/MatMul:product:0BQRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
(QRnnNetwork/EncodingNetwork/dense_1/SeluSelu4QRnnNetwork/EncodingNetwork/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
?QRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
AQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
AQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Љ
9QRnnNetwork/EncodingNetwork/batch_unflatten/strided_sliceStridedSlice8QRnnNetwork/EncodingNetwork/batch_flatten/Shape:output:0HQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack:output:0JQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_1:output:0JQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_maskЕ
1QRnnNetwork/EncodingNetwork/batch_unflatten/ShapeShape6QRnnNetwork/EncodingNetwork/dense_1/Selu:activations:0*
T0*
_output_shapes
:*
out_type0	:эа
AQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
CQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
CQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Б
;QRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1StridedSlice:QRnnNetwork/EncodingNetwork/batch_unflatten/Shape:output:0JQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack:output:0LQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_1:output:0LQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
end_masky
7QRnnNetwork/EncodingNetwork/batch_unflatten/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : И
2QRnnNetwork/EncodingNetwork/batch_unflatten/concatConcatV2BQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice:output:0DQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1:output:0@QRnnNetwork/EncodingNetwork/batch_unflatten/concat/axis:output:0*
N*
T0	*
_output_shapes
:ј
3QRnnNetwork/EncodingNetwork/batch_unflatten/ReshapeReshape6QRnnNetwork/EncodingNetwork/dense_1/Selu:activations:0;QRnnNetwork/EncodingNetwork/batch_unflatten/concat:output:0*
Tshape0	*
T0*,
_output_shapes
:џџџџџџџџџT
QRnnNetwork/mask/yConst*
_output_shapes
: *
dtype0*
value	B : 
QRnnNetwork/maskEqual!QRnnNetwork/ExpandDims_1:output:0QRnnNetwork/mask/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџa
QRnnNetwork/dynamic_unroll/RankConst*
_output_shapes
: *
dtype0*
value	B :h
&QRnnNetwork/dynamic_unroll/range/startConst*
_output_shapes
: *
dtype0*
value	B :h
&QRnnNetwork/dynamic_unroll/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :б
 QRnnNetwork/dynamic_unroll/rangeRange/QRnnNetwork/dynamic_unroll/range/start:output:0(QRnnNetwork/dynamic_unroll/Rank:output:0/QRnnNetwork/dynamic_unroll/range/delta:output:0*
_output_shapes
:{
*QRnnNetwork/dynamic_unroll/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB"       h
&QRnnNetwork/dynamic_unroll/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ь
!QRnnNetwork/dynamic_unroll/concatConcatV23QRnnNetwork/dynamic_unroll/concat/values_0:output:0)QRnnNetwork/dynamic_unroll/range:output:0/QRnnNetwork/dynamic_unroll/concat/axis:output:0*
N*
T0*
_output_shapes
:в
$QRnnNetwork/dynamic_unroll/transpose	Transpose<QRnnNetwork/EncodingNetwork/batch_unflatten/Reshape:output:0*QRnnNetwork/dynamic_unroll/concat:output:0*
T0*,
_output_shapes
:џџџџџџџџџ
 QRnnNetwork/dynamic_unroll/ShapeShape(QRnnNetwork/dynamic_unroll/transpose:y:0*
T0*
_output_shapes
::эЯx
.QRnnNetwork/dynamic_unroll/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:z
0QRnnNetwork/dynamic_unroll/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0QRnnNetwork/dynamic_unroll/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:и
(QRnnNetwork/dynamic_unroll/strided_sliceStridedSlice)QRnnNetwork/dynamic_unroll/Shape:output:07QRnnNetwork/dynamic_unroll/strided_slice/stack:output:09QRnnNetwork/dynamic_unroll/strided_slice/stack_1:output:09QRnnNetwork/dynamic_unroll/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask|
+QRnnNetwork/dynamic_unroll/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       Б
&QRnnNetwork/dynamic_unroll/transpose_1	TransposeQRnnNetwork/mask:z:04QRnnNetwork/dynamic_unroll/transpose_1/perm:output:0*
T0
*'
_output_shapes
:џџџџџџџџџk
)QRnnNetwork/dynamic_unroll/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :MФ
'QRnnNetwork/dynamic_unroll/zeros/packedPack1QRnnNetwork/dynamic_unroll/strided_slice:output:02QRnnNetwork/dynamic_unroll/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:k
&QRnnNetwork/dynamic_unroll/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Н
 QRnnNetwork/dynamic_unroll/zerosFill0QRnnNetwork/dynamic_unroll/zeros/packed:output:0/QRnnNetwork/dynamic_unroll/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџMm
+QRnnNetwork/dynamic_unroll/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :MШ
)QRnnNetwork/dynamic_unroll/zeros_1/packedPack1QRnnNetwork/dynamic_unroll/strided_slice:output:04QRnnNetwork/dynamic_unroll/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:m
(QRnnNetwork/dynamic_unroll/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    У
"QRnnNetwork/dynamic_unroll/zeros_1Fill2QRnnNetwork/dynamic_unroll/zeros_1/packed:output:01QRnnNetwork/dynamic_unroll/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџMЁ
"QRnnNetwork/dynamic_unroll/SqueezeSqueeze(QRnnNetwork/dynamic_unroll/transpose:y:0*
T0*(
_output_shapes
:џџџџџџџџџ*
squeeze_dims
  
$QRnnNetwork/dynamic_unroll/Squeeze_1Squeeze*QRnnNetwork/dynamic_unroll/transpose_1:y:0*
T0
*#
_output_shapes
:џџџџџџџџџ*
squeeze_dims
 г
!QRnnNetwork/dynamic_unroll/SelectSelect-QRnnNetwork/dynamic_unroll/Squeeze_1:output:0)QRnnNetwork/dynamic_unroll/zeros:output:0PartitionedCall_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџMз
#QRnnNetwork/dynamic_unroll/Select_1Select-QRnnNetwork/dynamic_unroll/Squeeze_1:output:0+QRnnNetwork/dynamic_unroll/zeros_1:output:0PartitionedCall_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџMР
:QRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOpReadVariableOpCqrnnnetwork_dynamic_unroll_lstm_cell_matmul_readvariableop_resource* 
_output_shapes
:
Д*
dtype0й
+QRnnNetwork/dynamic_unroll/lstm_cell/MatMulMatMul+QRnnNetwork/dynamic_unroll/Squeeze:output:0BQRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџДУ
<QRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpEqrnnnetwork_dynamic_unroll_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes
:	MД*
dtype0м
-QRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1MatMul*QRnnNetwork/dynamic_unroll/Select:output:0DQRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџДд
(QRnnNetwork/dynamic_unroll/lstm_cell/addAddV25QRnnNetwork/dynamic_unroll/lstm_cell/MatMul:product:07QRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџДН
;QRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOpReadVariableOpDqrnnnetwork_dynamic_unroll_lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:Д*
dtype0н
,QRnnNetwork/dynamic_unroll/lstm_cell/BiasAddBiasAdd,QRnnNetwork/dynamic_unroll/lstm_cell/add:z:0CQRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџДv
4QRnnNetwork/dynamic_unroll/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ѕ
*QRnnNetwork/dynamic_unroll/lstm_cell/splitSplit=QRnnNetwork/dynamic_unroll/lstm_cell/split/split_dim:output:05QRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџM:џџџџџџџџџM:џџџџџџџџџM:џџџџџџџџџM*
	num_split
,QRnnNetwork/dynamic_unroll/lstm_cell/SigmoidSigmoid3QRnnNetwork/dynamic_unroll/lstm_cell/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџM 
.QRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_1Sigmoid3QRnnNetwork/dynamic_unroll/lstm_cell/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџMУ
(QRnnNetwork/dynamic_unroll/lstm_cell/mulMul2QRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_1:y:0,QRnnNetwork/dynamic_unroll/Select_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџM
)QRnnNetwork/dynamic_unroll/lstm_cell/TanhTanh3QRnnNetwork/dynamic_unroll/lstm_cell/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџMФ
*QRnnNetwork/dynamic_unroll/lstm_cell/mul_1Mul0QRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid:y:0-QRnnNetwork/dynamic_unroll/lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџMУ
*QRnnNetwork/dynamic_unroll/lstm_cell/add_1AddV2,QRnnNetwork/dynamic_unroll/lstm_cell/mul:z:0.QRnnNetwork/dynamic_unroll/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџM 
.QRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_2Sigmoid3QRnnNetwork/dynamic_unroll/lstm_cell/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџM
+QRnnNetwork/dynamic_unroll/lstm_cell/Tanh_1Tanh.QRnnNetwork/dynamic_unroll/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџMШ
*QRnnNetwork/dynamic_unroll/lstm_cell/mul_2Mul2QRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_2:y:0/QRnnNetwork/dynamic_unroll/lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџMk
)QRnnNetwork/dynamic_unroll/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :Э
%QRnnNetwork/dynamic_unroll/ExpandDims
ExpandDims.QRnnNetwork/dynamic_unroll/lstm_cell/mul_2:z:02QRnnNetwork/dynamic_unroll/ExpandDims/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџMЃ
,QRnnNetwork/dense_2/Tensordot/ReadVariableOpReadVariableOp5qrnnnetwork_dense_2_tensordot_readvariableop_resource*
_output_shapes
:	Mи*
dtype0l
"QRnnNetwork/dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:s
"QRnnNetwork/dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       
#QRnnNetwork/dense_2/Tensordot/ShapeShape.QRnnNetwork/dynamic_unroll/ExpandDims:output:0*
T0*
_output_shapes
::эЯm
+QRnnNetwork/dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
&QRnnNetwork/dense_2/Tensordot/GatherV2GatherV2,QRnnNetwork/dense_2/Tensordot/Shape:output:0+QRnnNetwork/dense_2/Tensordot/free:output:04QRnnNetwork/dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:o
-QRnnNetwork/dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
(QRnnNetwork/dense_2/Tensordot/GatherV2_1GatherV2,QRnnNetwork/dense_2/Tensordot/Shape:output:0+QRnnNetwork/dense_2/Tensordot/axes:output:06QRnnNetwork/dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:m
#QRnnNetwork/dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: Њ
"QRnnNetwork/dense_2/Tensordot/ProdProd/QRnnNetwork/dense_2/Tensordot/GatherV2:output:0,QRnnNetwork/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: o
%QRnnNetwork/dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: А
$QRnnNetwork/dense_2/Tensordot/Prod_1Prod1QRnnNetwork/dense_2/Tensordot/GatherV2_1:output:0.QRnnNetwork/dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: k
)QRnnNetwork/dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ь
$QRnnNetwork/dense_2/Tensordot/concatConcatV2+QRnnNetwork/dense_2/Tensordot/free:output:0+QRnnNetwork/dense_2/Tensordot/axes:output:02QRnnNetwork/dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:Е
#QRnnNetwork/dense_2/Tensordot/stackPack+QRnnNetwork/dense_2/Tensordot/Prod:output:0-QRnnNetwork/dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Щ
'QRnnNetwork/dense_2/Tensordot/transpose	Transpose.QRnnNetwork/dynamic_unroll/ExpandDims:output:0-QRnnNetwork/dense_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџMЦ
%QRnnNetwork/dense_2/Tensordot/ReshapeReshape+QRnnNetwork/dense_2/Tensordot/transpose:y:0,QRnnNetwork/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџЧ
$QRnnNetwork/dense_2/Tensordot/MatMulMatMul.QRnnNetwork/dense_2/Tensordot/Reshape:output:04QRnnNetwork/dense_2/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџиp
%QRnnNetwork/dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:иm
+QRnnNetwork/dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ї
&QRnnNetwork/dense_2/Tensordot/concat_1ConcatV2/QRnnNetwork/dense_2/Tensordot/GatherV2:output:0.QRnnNetwork/dense_2/Tensordot/Const_2:output:04QRnnNetwork/dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Р
QRnnNetwork/dense_2/TensordotReshape.QRnnNetwork/dense_2/Tensordot/MatMul:product:0/QRnnNetwork/dense_2/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:џџџџџџџџџи
*QRnnNetwork/dense_2/BiasAdd/ReadVariableOpReadVariableOp3qrnnnetwork_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:и*
dtype0Й
QRnnNetwork/dense_2/BiasAddBiasAdd&QRnnNetwork/dense_2/Tensordot:output:02QRnnNetwork/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџи}
QRnnNetwork/dense_2/SeluSelu$QRnnNetwork/dense_2/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџиЃ
,QRnnNetwork/dense_3/Tensordot/ReadVariableOpReadVariableOp5qrnnnetwork_dense_3_tensordot_readvariableop_resource*
_output_shapes
:	иd*
dtype0l
"QRnnNetwork/dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:s
"QRnnNetwork/dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       
#QRnnNetwork/dense_3/Tensordot/ShapeShape&QRnnNetwork/dense_2/Selu:activations:0*
T0*
_output_shapes
::эЯm
+QRnnNetwork/dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
&QRnnNetwork/dense_3/Tensordot/GatherV2GatherV2,QRnnNetwork/dense_3/Tensordot/Shape:output:0+QRnnNetwork/dense_3/Tensordot/free:output:04QRnnNetwork/dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:o
-QRnnNetwork/dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
(QRnnNetwork/dense_3/Tensordot/GatherV2_1GatherV2,QRnnNetwork/dense_3/Tensordot/Shape:output:0+QRnnNetwork/dense_3/Tensordot/axes:output:06QRnnNetwork/dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:m
#QRnnNetwork/dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: Њ
"QRnnNetwork/dense_3/Tensordot/ProdProd/QRnnNetwork/dense_3/Tensordot/GatherV2:output:0,QRnnNetwork/dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: o
%QRnnNetwork/dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: А
$QRnnNetwork/dense_3/Tensordot/Prod_1Prod1QRnnNetwork/dense_3/Tensordot/GatherV2_1:output:0.QRnnNetwork/dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: k
)QRnnNetwork/dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ь
$QRnnNetwork/dense_3/Tensordot/concatConcatV2+QRnnNetwork/dense_3/Tensordot/free:output:0+QRnnNetwork/dense_3/Tensordot/axes:output:02QRnnNetwork/dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:Е
#QRnnNetwork/dense_3/Tensordot/stackPack+QRnnNetwork/dense_3/Tensordot/Prod:output:0-QRnnNetwork/dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Т
'QRnnNetwork/dense_3/Tensordot/transpose	Transpose&QRnnNetwork/dense_2/Selu:activations:0-QRnnNetwork/dense_3/Tensordot/concat:output:0*
T0*,
_output_shapes
:џџџџџџџџџиЦ
%QRnnNetwork/dense_3/Tensordot/ReshapeReshape+QRnnNetwork/dense_3/Tensordot/transpose:y:0,QRnnNetwork/dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџЦ
$QRnnNetwork/dense_3/Tensordot/MatMulMatMul.QRnnNetwork/dense_3/Tensordot/Reshape:output:04QRnnNetwork/dense_3/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџdo
%QRnnNetwork/dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:dm
+QRnnNetwork/dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ї
&QRnnNetwork/dense_3/Tensordot/concat_1ConcatV2/QRnnNetwork/dense_3/Tensordot/GatherV2:output:0.QRnnNetwork/dense_3/Tensordot/Const_2:output:04QRnnNetwork/dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:П
QRnnNetwork/dense_3/TensordotReshape.QRnnNetwork/dense_3/Tensordot/MatMul:product:0/QRnnNetwork/dense_3/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџd
*QRnnNetwork/dense_3/BiasAdd/ReadVariableOpReadVariableOp3qrnnnetwork_dense_3_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0И
QRnnNetwork/dense_3/BiasAddBiasAdd&QRnnNetwork/dense_3/Tensordot:output:02QRnnNetwork/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџd|
QRnnNetwork/dense_3/SeluSelu$QRnnNetwork/dense_3/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџdФ
=QRnnNetwork/num_action_project/dense/Tensordot/ReadVariableOpReadVariableOpFqrnnnetwork_num_action_project_dense_tensordot_readvariableop_resource*
_output_shapes

:d*
dtype0}
3QRnnNetwork/num_action_project/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
3QRnnNetwork/num_action_project/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       
4QRnnNetwork/num_action_project/dense/Tensordot/ShapeShape&QRnnNetwork/dense_3/Selu:activations:0*
T0*
_output_shapes
::эЯ~
<QRnnNetwork/num_action_project/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Я
7QRnnNetwork/num_action_project/dense/Tensordot/GatherV2GatherV2=QRnnNetwork/num_action_project/dense/Tensordot/Shape:output:0<QRnnNetwork/num_action_project/dense/Tensordot/free:output:0EQRnnNetwork/num_action_project/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
>QRnnNetwork/num_action_project/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : г
9QRnnNetwork/num_action_project/dense/Tensordot/GatherV2_1GatherV2=QRnnNetwork/num_action_project/dense/Tensordot/Shape:output:0<QRnnNetwork/num_action_project/dense/Tensordot/axes:output:0GQRnnNetwork/num_action_project/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:~
4QRnnNetwork/num_action_project/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: н
3QRnnNetwork/num_action_project/dense/Tensordot/ProdProd@QRnnNetwork/num_action_project/dense/Tensordot/GatherV2:output:0=QRnnNetwork/num_action_project/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 
6QRnnNetwork/num_action_project/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: у
5QRnnNetwork/num_action_project/dense/Tensordot/Prod_1ProdBQRnnNetwork/num_action_project/dense/Tensordot/GatherV2_1:output:0?QRnnNetwork/num_action_project/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: |
:QRnnNetwork/num_action_project/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : А
5QRnnNetwork/num_action_project/dense/Tensordot/concatConcatV2<QRnnNetwork/num_action_project/dense/Tensordot/free:output:0<QRnnNetwork/num_action_project/dense/Tensordot/axes:output:0CQRnnNetwork/num_action_project/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:ш
4QRnnNetwork/num_action_project/dense/Tensordot/stackPack<QRnnNetwork/num_action_project/dense/Tensordot/Prod:output:0>QRnnNetwork/num_action_project/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:у
8QRnnNetwork/num_action_project/dense/Tensordot/transpose	Transpose&QRnnNetwork/dense_3/Selu:activations:0>QRnnNetwork/num_action_project/dense/Tensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџdљ
6QRnnNetwork/num_action_project/dense/Tensordot/ReshapeReshape<QRnnNetwork/num_action_project/dense/Tensordot/transpose:y:0=QRnnNetwork/num_action_project/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџљ
5QRnnNetwork/num_action_project/dense/Tensordot/MatMulMatMul?QRnnNetwork/num_action_project/dense/Tensordot/Reshape:output:0EQRnnNetwork/num_action_project/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
6QRnnNetwork/num_action_project/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:~
<QRnnNetwork/num_action_project/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Л
7QRnnNetwork/num_action_project/dense/Tensordot/concat_1ConcatV2@QRnnNetwork/num_action_project/dense/Tensordot/GatherV2:output:0?QRnnNetwork/num_action_project/dense/Tensordot/Const_2:output:0EQRnnNetwork/num_action_project/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:ђ
.QRnnNetwork/num_action_project/dense/TensordotReshape?QRnnNetwork/num_action_project/dense/Tensordot/MatMul:product:0@QRnnNetwork/num_action_project/dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџМ
;QRnnNetwork/num_action_project/dense/BiasAdd/ReadVariableOpReadVariableOpDqrnnnetwork_num_action_project_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ы
,QRnnNetwork/num_action_project/dense/BiasAddBiasAdd7QRnnNetwork/num_action_project/dense/Tensordot:output:0CQRnnNetwork/num_action_project/dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ
QRnnNetwork/SqueezeSqueeze5QRnnNetwork/num_action_project/dense/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
squeeze_dims
l
!Categorical/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
Categorical/mode/ArgMaxArgMaxQRnnNetwork/Squeeze:output:0*Categorical/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:џџџџџџџџџ|
Categorical/mode/CastCast Categorical/mode/ArgMax:output:0*

DstT0*

SrcT0	*#
_output_shapes
:џџџџџџџџџW
Deterministic/atolConst*
_output_shapes
: *
dtype0*
valueB
 *    W
Deterministic/rtolConst*
_output_shapes
: *
dtype0*
valueB
 *    d
!Deterministic/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB q
Deterministic/sample/ShapeShapeCategorical/mode/Cast:y:0*
T0*
_output_shapes
::эЯ\
Deterministic/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : r
(Deterministic/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*Deterministic/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*Deterministic/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:И
"Deterministic/sample/strided_sliceStridedSlice#Deterministic/sample/Shape:output:01Deterministic/sample/strided_slice/stack:output:03Deterministic/sample/strided_slice/stack_1:output:03Deterministic/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskh
%Deterministic/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB j
'Deterministic/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB Ў
"Deterministic/sample/BroadcastArgsBroadcastArgs0Deterministic/sample/BroadcastArgs/s0_1:output:0+Deterministic/sample/strided_slice:output:0*
_output_shapes
:n
$Deterministic/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:g
$Deterministic/sample/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB b
 Deterministic/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Deterministic/sample/concatConcatV2-Deterministic/sample/concat/values_0:output:0'Deterministic/sample/BroadcastArgs:r0:0-Deterministic/sample/concat/values_2:output:0)Deterministic/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:Ђ
 Deterministic/sample/BroadcastToBroadcastToCategorical/mode/Cast:y:0$Deterministic/sample/concat:output:0*
T0*'
_output_shapes
:џџџџџџџџџe
"Deterministic/sample/Shape_1/ConstConst*
_output_shapes
: *
dtype0*
valueB f
Deterministic/sample/Shape_1Const*
_output_shapes
:*
dtype0*
valueB: 
Deterministic/sample/Shape_2Shape)Deterministic/sample/BroadcastTo:output:0*
T0*
_output_shapes
::эЯt
*Deterministic/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:v
,Deterministic/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,Deterministic/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Р
$Deterministic/sample/strided_slice_1StridedSlice%Deterministic/sample/Shape_2:output:03Deterministic/sample/strided_slice_1/stack:output:05Deterministic/sample/strided_slice_1/stack_1:output:05Deterministic/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskd
"Deterministic/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : п
Deterministic/sample/concat_1ConcatV2*Deterministic/sample/sample_shape:output:0-Deterministic/sample/strided_slice_1:output:0+Deterministic/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Ј
Deterministic/sample/ReshapeReshape)Deterministic/sample/BroadcastTo:output:0&Deterministic/sample/concat_1:output:0*
T0*#
_output_shapes
:џџџџџџџџџ\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @
clip_by_value/MinimumMinimum%Deterministic/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0*#
_output_shapes
:џџџџџџџџџT
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    {
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*#
_output_shapes
:џџџџџџџџџ\
IdentityIdentityclip_by_value:z:0^NoOp*
T0*#
_output_shapes
:џџџџџџџџџ

Identity_1Identity.QRnnNetwork/dynamic_unroll/lstm_cell/mul_2:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџM

Identity_2Identity.QRnnNetwork/dynamic_unroll/lstm_cell/add_1:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџM
NoOpNoOp9^QRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp8^QRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp;^QRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp:^QRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp+^QRnnNetwork/dense_2/BiasAdd/ReadVariableOp-^QRnnNetwork/dense_2/Tensordot/ReadVariableOp+^QRnnNetwork/dense_3/BiasAdd/ReadVariableOp-^QRnnNetwork/dense_3/Tensordot/ReadVariableOp<^QRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOp;^QRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOp=^QRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOp<^QRnnNetwork/num_action_project/dense/BiasAdd/ReadVariableOp>^QRnnNetwork/num_action_project/dense/Tensordot/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџM:џџџџџџџџџM: : : : : : : : : : : : : 2t
8QRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp8QRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp2r
7QRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp7QRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp2x
:QRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp:QRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp2v
9QRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp9QRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp2X
*QRnnNetwork/dense_2/BiasAdd/ReadVariableOp*QRnnNetwork/dense_2/BiasAdd/ReadVariableOp2\
,QRnnNetwork/dense_2/Tensordot/ReadVariableOp,QRnnNetwork/dense_2/Tensordot/ReadVariableOp2X
*QRnnNetwork/dense_3/BiasAdd/ReadVariableOp*QRnnNetwork/dense_3/BiasAdd/ReadVariableOp2\
,QRnnNetwork/dense_3/Tensordot/ReadVariableOp,QRnnNetwork/dense_3/Tensordot/ReadVariableOp2z
;QRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOp;QRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOp2x
:QRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOp:QRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOp2|
<QRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOp<QRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOp2z
;QRnnNetwork/num_action_project/dense/BiasAdd/ReadVariableOp;QRnnNetwork/num_action_project/dense/BiasAdd/ReadVariableOp2~
=QRnnNetwork/num_action_project/dense/Tensordot/ReadVariableOp=QRnnNetwork/num_action_project/dense/Tensordot/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:UQ
'
_output_shapes
:џџџџџџџџџM
&
_user_specified_namepolicy_state:UQ
'
_output_shapes
:џџџџџџџџџM
&
_user_specified_namepolicy_state:RN
'
_output_shapes
:џџџџџџџџџ
#
_user_specified_name	time_step:NJ
#
_output_shapes
:џџџџџџџџџ
#
_user_specified_name	time_step:NJ
#
_output_shapes
:џџџџџџџџџ
#
_user_specified_name	time_step:N J
#
_output_shapes
:џџџџџџџџџ
#
_user_specified_name	time_step
§
T
$__inference_get_initial_state_294241

batch_size
identity

identity_1H
packedPack
batch_size*
N*
T0*
_output_shapes
:Y
shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:MM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
concatConcatV2packed:output:0shape_as_tensor:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    f
zerosFillconcat:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџM[
shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB:MO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
concat_1ConcatV2packed:output:0shape_as_tensor_1:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zeros_1Fillconcat_1:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџMV
IdentityIdentityzeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџMZ

Identity_1Identityzeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџM"!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
Ъ
_
"__inference_per_field_where_294909
t
f
sub_rank
shape_equal

identityY
assert_rank_at_least/ShapeShapet*
T0*
_output_shapes
::эЯa
Cassert_rank_at_least/assert_type/statically_determined_correct_typeNoOp*
_output_shapes
 R
4assert_rank_at_least/static_checks_determined_all_okNoOp*
_output_shapes
 F
RankConst*
_output_shapes
: *
dtype0*
value	B :D
subSubRank:output:0sub_rank*
T0*
_output_shapes
: N
ShapeShapeshape_equal*
T0
*
_output_shapes
::эЯe
ones/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџb
ones/ReshapeReshapesub:z:0ones/Reshape/shape:output:0*
T0*
_output_shapes
:L

ones/ConstConst*
_output_shapes
: *
dtype0*
value	B :f
onesFillones/Reshape:output:0ones/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ~
concatConcatV2Shape:output:0ones:output:0concat/axis:output:0*
N*
T0*#
_output_shapes
:џџџџџџџџџS
ReshapeReshapeshape_equalconcat:output:0*
T0
*
_output_shapes
:O
SelectV2SelectV2Reshape:output:0tf*
T0*
_output_shapes
:J
IdentityIdentitySelectV2:output:0*
T0*
_output_shapes
:"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:џџџџџџџџџM:џџџџџџџџџM: :џџџџџџџџџ:JF
#
_output_shapes
:џџџџџџџџџ

_user_specified_nameEqual:<8

_output_shapes
: 

_user_specified_nameRank:JF
'
_output_shapes
:џџџџџџџџџM

_user_specified_namef:J F
'
_output_shapes
:џџџџџџџџџM

_user_specified_namet
и
c
"__inference_per_field_where_294357
t
f

sub_rank_1
shape_equal_1

identityY
assert_rank_at_least/ShapeShapet*
T0*
_output_shapes
::эЯa
Cassert_rank_at_least/assert_type/statically_determined_correct_typeNoOp*
_output_shapes
 R
4assert_rank_at_least/static_checks_determined_all_okNoOp*
_output_shapes
 F
RankConst*
_output_shapes
: *
dtype0*
value	B :F
subSubRank:output:0
sub_rank_1*
T0*
_output_shapes
: P
ShapeShapeshape_equal_1*
T0
*
_output_shapes
::эЯe
ones/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџb
ones/ReshapeReshapesub:z:0ones/Reshape/shape:output:0*
T0*
_output_shapes
:L

ones/ConstConst*
_output_shapes
: *
dtype0*
value	B :f
onesFillones/Reshape:output:0ones/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ~
concatConcatV2Shape:output:0ones:output:0concat/axis:output:0*
N*
T0*#
_output_shapes
:џџџџџџџџџU
ReshapeReshapeshape_equal_1concat:output:0*
T0
*
_output_shapes
:O
SelectV2SelectV2Reshape:output:0tf*
T0*
_output_shapes
:J
IdentityIdentitySelectV2:output:0*
T0*
_output_shapes
:"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:џџџџџџџџџM:џџџџџџџџџM: :џџџџџџџџџ:LH
#
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	Equal_1:>:

_output_shapes
: 
 
_user_specified_nameRank_1:JF
'
_output_shapes
:џџџџџџџџџM

_user_specified_namef:J F
'
_output_shapes
:џџџџџџџџџM

_user_specified_namet
Ъ
_
"__inference_per_field_where_294613
t
f
sub_rank
shape_equal

identityY
assert_rank_at_least/ShapeShapet*
T0*
_output_shapes
::эЯa
Cassert_rank_at_least/assert_type/statically_determined_correct_typeNoOp*
_output_shapes
 R
4assert_rank_at_least/static_checks_determined_all_okNoOp*
_output_shapes
 F
RankConst*
_output_shapes
: *
dtype0*
value	B :D
subSubRank:output:0sub_rank*
T0*
_output_shapes
: N
ShapeShapeshape_equal*
T0
*
_output_shapes
::эЯe
ones/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџb
ones/ReshapeReshapesub:z:0ones/Reshape/shape:output:0*
T0*
_output_shapes
:L

ones/ConstConst*
_output_shapes
: *
dtype0*
value	B :f
onesFillones/Reshape:output:0ones/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ~
concatConcatV2Shape:output:0ones:output:0concat/axis:output:0*
N*
T0*#
_output_shapes
:џџџџџџџџџS
ReshapeReshapeshape_equalconcat:output:0*
T0
*
_output_shapes
:O
SelectV2SelectV2Reshape:output:0tf*
T0*
_output_shapes
:J
IdentityIdentitySelectV2:output:0*
T0*
_output_shapes
:"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:џџџџџџџџџM:џџџџџџџџџM: :џџџџџџџџџ:JF
#
_output_shapes
:џџџџџџџџџ

_user_specified_nameEqual:<8

_output_shapes
: 

_user_specified_nameRank:JF
'
_output_shapes
:џџџџџџџџџM

_user_specified_namef:J F
'
_output_shapes
:џџџџџџџџџM

_user_specified_namet
и
c
"__inference_per_field_where_294653
t
f

sub_rank_1
shape_equal_1

identityY
assert_rank_at_least/ShapeShapet*
T0*
_output_shapes
::эЯa
Cassert_rank_at_least/assert_type/statically_determined_correct_typeNoOp*
_output_shapes
 R
4assert_rank_at_least/static_checks_determined_all_okNoOp*
_output_shapes
 F
RankConst*
_output_shapes
: *
dtype0*
value	B :F
subSubRank:output:0
sub_rank_1*
T0*
_output_shapes
: P
ShapeShapeshape_equal_1*
T0
*
_output_shapes
::эЯe
ones/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџb
ones/ReshapeReshapesub:z:0ones/Reshape/shape:output:0*
T0*
_output_shapes
:L

ones/ConstConst*
_output_shapes
: *
dtype0*
value	B :f
onesFillones/Reshape:output:0ones/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ~
concatConcatV2Shape:output:0ones:output:0concat/axis:output:0*
N*
T0*#
_output_shapes
:џџџџџџџџџU
ReshapeReshapeshape_equal_1concat:output:0*
T0
*
_output_shapes
:O
SelectV2SelectV2Reshape:output:0tf*
T0*
_output_shapes
:J
IdentityIdentitySelectV2:output:0*
T0*
_output_shapes
:"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:џџџџџџџџџM:џџџџџџџџџM: :џџџџџџџџџ:LH
#
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	Equal_1:>:

_output_shapes
: 
 
_user_specified_nameRank_1:JF
'
_output_shapes
:џџџџџџџџџM

_user_specified_namef:J F
'
_output_shapes
:џџџџџџџџџM

_user_specified_namet
ЂI
Ь

"__inference__traced_restore_295317
file_prefix#
assignvariableop_variable:	 N
;assignvariableop_1_qrnnnetwork_encodingnetwork_dense_kernel:	бH
9assignvariableop_2_qrnnnetwork_encodingnetwork_dense_bias:	бQ
=assignvariableop_3_qrnnnetwork_encodingnetwork_dense_1_kernel:
бJ
;assignvariableop_4_qrnnnetwork_encodingnetwork_dense_1_bias:	H
4assignvariableop_5_qrnnnetwork_dynamic_unroll_kernel:
ДQ
>assignvariableop_6_qrnnnetwork_dynamic_unroll_recurrent_kernel:	MДA
2assignvariableop_7_qrnnnetwork_dynamic_unroll_bias:	Д@
-assignvariableop_8_qrnnnetwork_dense_2_kernel:	Mи:
+assignvariableop_9_qrnnnetwork_dense_2_bias:	иA
.assignvariableop_10_qrnnnetwork_dense_3_kernel:	иd:
,assignvariableop_11_qrnnnetwork_dense_3_bias:dQ
?assignvariableop_12_qrnnnetwork_num_action_project_dense_kernel:dK
=assignvariableop_13_qrnnnetwork_num_action_project_dense_bias:
identity_15ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_2ЂAssignVariableOp_3ЂAssignVariableOp_4ЂAssignVariableOp_5ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Г
valueЉBІB%train_step/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/0/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/1/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/2/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/3/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/4/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/5/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/6/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/7/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/8/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/9/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/10/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/11/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/12/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B B B B B B B B B B B B B B B щ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*P
_output_shapes>
<:::::::::::::::*
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:Ќ
AssignVariableOpAssignVariableOpassignvariableop_variableIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:в
AssignVariableOp_1AssignVariableOp;assignvariableop_1_qrnnnetwork_encodingnetwork_dense_kernelIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:а
AssignVariableOp_2AssignVariableOp9assignvariableop_2_qrnnnetwork_encodingnetwork_dense_biasIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:д
AssignVariableOp_3AssignVariableOp=assignvariableop_3_qrnnnetwork_encodingnetwork_dense_1_kernelIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:в
AssignVariableOp_4AssignVariableOp;assignvariableop_4_qrnnnetwork_encodingnetwork_dense_1_biasIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_5AssignVariableOp4assignvariableop_5_qrnnnetwork_dynamic_unroll_kernelIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:е
AssignVariableOp_6AssignVariableOp>assignvariableop_6_qrnnnetwork_dynamic_unroll_recurrent_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:Щ
AssignVariableOp_7AssignVariableOp2assignvariableop_7_qrnnnetwork_dynamic_unroll_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_8AssignVariableOp-assignvariableop_8_qrnnnetwork_dense_2_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_9AssignVariableOp+assignvariableop_9_qrnnnetwork_dense_2_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_10AssignVariableOp.assignvariableop_10_qrnnnetwork_dense_3_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_11AssignVariableOp,assignvariableop_11_qrnnnetwork_dense_3_biasIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:и
AssignVariableOp_12AssignVariableOp?assignvariableop_12_qrnnnetwork_num_action_project_dense_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:ж
AssignVariableOp_13AssignVariableOp=assignvariableop_13_qrnnnetwork_num_action_project_dense_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 
Identity_14Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_15IdentityIdentity_14:output:0^NoOp_1*
T0*
_output_shapes
: Ь
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_15Identity_15:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
: : : : : : : : : : : : : : : 2*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:IE
C
_user_specified_name+)QRnnNetwork/num_action_project/dense/bias:KG
E
_user_specified_name-+QRnnNetwork/num_action_project/dense/kernel:84
2
_user_specified_nameQRnnNetwork/dense_3/bias::6
4
_user_specified_nameQRnnNetwork/dense_3/kernel:8
4
2
_user_specified_nameQRnnNetwork/dense_2/bias::	6
4
_user_specified_nameQRnnNetwork/dense_2/kernel:?;
9
_user_specified_name!QRnnNetwork/dynamic_unroll/bias:KG
E
_user_specified_name-+QRnnNetwork/dynamic_unroll/recurrent_kernel:A=
;
_user_specified_name#!QRnnNetwork/dynamic_unroll/kernel:HD
B
_user_specified_name*(QRnnNetwork/EncodingNetwork/dense_1/bias:JF
D
_user_specified_name,*QRnnNetwork/EncodingNetwork/dense_1/kernel:FB
@
_user_specified_name(&QRnnNetwork/EncodingNetwork/dense/bias:HD
B
_user_specified_name*(QRnnNetwork/EncodingNetwork/dense/kernel:($
"
_user_specified_name
Variable:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ъ
_
"__inference_per_field_where_294317
t
f
sub_rank
shape_equal

identityY
assert_rank_at_least/ShapeShapet*
T0*
_output_shapes
::эЯa
Cassert_rank_at_least/assert_type/statically_determined_correct_typeNoOp*
_output_shapes
 R
4assert_rank_at_least/static_checks_determined_all_okNoOp*
_output_shapes
 F
RankConst*
_output_shapes
: *
dtype0*
value	B :D
subSubRank:output:0sub_rank*
T0*
_output_shapes
: N
ShapeShapeshape_equal*
T0
*
_output_shapes
::эЯe
ones/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџb
ones/ReshapeReshapesub:z:0ones/Reshape/shape:output:0*
T0*
_output_shapes
:L

ones/ConstConst*
_output_shapes
: *
dtype0*
value	B :f
onesFillones/Reshape:output:0ones/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ~
concatConcatV2Shape:output:0ones:output:0concat/axis:output:0*
N*
T0*#
_output_shapes
:џџџџџџџџџS
ReshapeReshapeshape_equalconcat:output:0*
T0
*
_output_shapes
:O
SelectV2SelectV2Reshape:output:0tf*
T0*
_output_shapes
:J
IdentityIdentitySelectV2:output:0*
T0*
_output_shapes
:"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:џџџџџџџџџM:џџџџџџџџџM: :џџџџџџџџџ:JF
#
_output_shapes
:џџџџџџџџџ

_user_specified_nameEqual:<8

_output_shapes
: 

_user_specified_nameRank:JF
'
_output_shapes
:џџџџџџџџџM

_user_specified_namef:J F
'
_output_shapes
:џџџџџџџџџM

_user_specified_namet
У
О
*__inference_function_with_signature_294182
	step_type

reward
discount
observation
unknown
	unknown_0
	unknown_1:	б
	unknown_2:	б
	unknown_3:
б
	unknown_4:	
	unknown_5:
Д
	unknown_6:	MД
	unknown_7:	Д
	unknown_8:	Mи
	unknown_9:	и

unknown_10:	иd

unknown_11:d

unknown_12:d

unknown_13:
identity

identity_1

identity_2ЂStatefulPartitionedCallЅ
StatefulPartitionedCallStatefulPartitionedCall	step_typerewarddiscountobservationunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13*
Tin
2*
Tout
2*
_collective_manager_ids
 *I
_output_shapes7
5:џџџџџџџџџ:џџџџџџџџџM:џџџџџџџџџM*/
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *"
fR
__inference_action_294149k
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*#
_output_shapes
:џџџџџџџџџq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:џџџџџџџџџMq

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:џџџџџџџџџM<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџM:џџџџџџџџџM: : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name294174:&"
 
_user_specified_name294172:&"
 
_user_specified_name294170:&"
 
_user_specified_name294168:&"
 
_user_specified_name294166:&"
 
_user_specified_name294164:&"
 
_user_specified_name294162:&"
 
_user_specified_name294160:&
"
 
_user_specified_name294158:&	"
 
_user_specified_name294156:&"
 
_user_specified_name294154:&"
 
_user_specified_name294152:&"
 
_user_specified_name294150:LH
'
_output_shapes
:џџџџџџџџџM

_user_specified_name1/1:LH
'
_output_shapes
:џџџџџџџџџM

_user_specified_name1/0:VR
'
_output_shapes
:џџџџџџџџџ
'
_user_specified_name0/observation:OK
#
_output_shapes
:џџџџџџџџџ
$
_user_specified_name
0/discount:MI
#
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
0/reward:P L
#
_output_shapes
:џџџџџџџџџ
%
_user_specified_name0/step_type

Z
*__inference_function_with_signature_294246

batch_size
identity

identity_1Њ
PartitionedCallPartitionedCall
batch_size*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:џџџџџџџџџM:џџџџџџџџџM* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *-
f(R&
$__inference_get_initial_state_294241`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџMb

Identity_1IdentityPartitionedCall:output:1*
T0*'
_output_shapes
:џџџџџџџџџM"!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
эв
Є
__inference_action_294568
	step_type

reward
discount
observation
unknown
	unknown_0S
@qrnnnetwork_encodingnetwork_dense_matmul_readvariableop_resource:	бP
Aqrnnnetwork_encodingnetwork_dense_biasadd_readvariableop_resource:	бV
Bqrnnnetwork_encodingnetwork_dense_1_matmul_readvariableop_resource:
бR
Cqrnnnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resource:	W
Cqrnnnetwork_dynamic_unroll_lstm_cell_matmul_readvariableop_resource:
ДX
Eqrnnnetwork_dynamic_unroll_lstm_cell_matmul_1_readvariableop_resource:	MДS
Dqrnnnetwork_dynamic_unroll_lstm_cell_biasadd_readvariableop_resource:	ДH
5qrnnnetwork_dense_2_tensordot_readvariableop_resource:	MиB
3qrnnnetwork_dense_2_biasadd_readvariableop_resource:	иH
5qrnnnetwork_dense_3_tensordot_readvariableop_resource:	иdA
3qrnnnetwork_dense_3_biasadd_readvariableop_resource:dX
Fqrnnnetwork_num_action_project_dense_tensordot_readvariableop_resource:dR
Dqrnnnetwork_num_action_project_dense_biasadd_readvariableop_resource:
identity

identity_1

identity_2Ђ8QRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpЂ7QRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpЂ:QRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpЂ9QRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpЂ*QRnnNetwork/dense_2/BiasAdd/ReadVariableOpЂ,QRnnNetwork/dense_2/Tensordot/ReadVariableOpЂ*QRnnNetwork/dense_3/BiasAdd/ReadVariableOpЂ,QRnnNetwork/dense_3/Tensordot/ReadVariableOpЂ;QRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOpЂ:QRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOpЂ<QRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOpЂ;QRnnNetwork/num_action_project/dense/BiasAdd/ReadVariableOpЂ=QRnnNetwork/num_action_project/dense/Tensordot/ReadVariableOpK
ShapeShapediscount*
T0*
_output_shapes
::эЯ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskT
packedPackstrided_slice:output:0*
N*
T0*
_output_shapes
:Y
shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:MM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
concatConcatV2packed:output:0shape_as_tensor:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    f
zerosFillconcat:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџM[
shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB:MO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
concat_1ConcatV2packed:output:0shape_as_tensor_1:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zeros_1Fillconcat_1:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџMI
Equal/yConst*
_output_shapes
: *
dtype0*
value	B : Y
EqualEqual	step_typeEqual/y:output:0*
T0*#
_output_shapes
:џџџџџџџџџF
RankConst*
_output_shapes
: *
dtype0*
value	B :О
PartitionedCallPartitionedCallzeros:output:0unknownRank:output:0	Equal:z:0*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџM* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference_per_field_where_294317Ф
PartitionedCall_1PartitionedCallzeros_1:output:0	unknown_0Rank:output:0	Equal:z:0*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџM* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference_per_field_where_294317M
Shape_1Shapediscount*
T0*
_output_shapes
::эЯ_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
packed_1Packstrided_slice_1:output:0*
N*
T0*
_output_shapes
:[
shape_as_tensor_2Const*
_output_shapes
:*
dtype0*
valueB:MO
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
concat_2ConcatV2packed_1:output:0shape_as_tensor_2:output:0concat_2/axis:output:0*
N*
T0*
_output_shapes
:R
zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zeros_2Fillconcat_2:output:0zeros_2/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџM[
shape_as_tensor_3Const*
_output_shapes
:*
dtype0*
valueB:MO
concat_3/axisConst*
_output_shapes
: *
dtype0*
value	B : 
concat_3ConcatV2packed_1:output:0shape_as_tensor_3:output:0concat_3/axis:output:0*
N*
T0*
_output_shapes
:R
zeros_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zeros_3Fillconcat_3:output:0zeros_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџMK
	Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : ]
Equal_1Equal	step_typeEqual_1/y:output:0*
T0*#
_output_shapes
:џџџџџџџџџH
Rank_1Const*
_output_shapes
: *
dtype0*
value	B :з
PartitionedCall_2PartitionedCallzeros_2:output:0PartitionedCall:output:0Rank_1:output:0Equal_1:z:0*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџM* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference_per_field_where_294357й
PartitionedCall_3PartitionedCallzeros_3:output:0PartitionedCall_1:output:0Rank_1:output:0Equal_1:z:0*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџM* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference_per_field_where_294357\
QRnnNetwork/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :
QRnnNetwork/ExpandDims
ExpandDimsobservation#QRnnNetwork/ExpandDims/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџ^
QRnnNetwork/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :
QRnnNetwork/ExpandDims_1
ExpandDims	step_type%QRnnNetwork/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
/QRnnNetwork/EncodingNetwork/batch_flatten/ShapeShapeQRnnNetwork/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	:эа
7QRnnNetwork/EncodingNetwork/batch_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   б
1QRnnNetwork/EncodingNetwork/batch_flatten/ReshapeReshapeQRnnNetwork/ExpandDims:output:0@QRnnNetwork/EncodingNetwork/batch_flatten/Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџz
)QRnnNetwork/EncodingNetwork/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   и
+QRnnNetwork/EncodingNetwork/flatten/ReshapeReshape:QRnnNetwork/EncodingNetwork/batch_flatten/Reshape:output:02QRnnNetwork/EncodingNetwork/flatten/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџЙ
7QRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpReadVariableOp@qrnnnetwork_encodingnetwork_dense_matmul_readvariableop_resource*
_output_shapes
:	б*
dtype0м
(QRnnNetwork/EncodingNetwork/dense/MatMulMatMul4QRnnNetwork/EncodingNetwork/flatten/Reshape:output:0?QRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџбЗ
8QRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpReadVariableOpAqrnnnetwork_encodingnetwork_dense_biasadd_readvariableop_resource*
_output_shapes	
:б*
dtype0н
)QRnnNetwork/EncodingNetwork/dense/BiasAddBiasAdd2QRnnNetwork/EncodingNetwork/dense/MatMul:product:0@QRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџб
&QRnnNetwork/EncodingNetwork/dense/SeluSelu2QRnnNetwork/EncodingNetwork/dense/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџбО
9QRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpReadVariableOpBqrnnnetwork_encodingnetwork_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
б*
dtype0р
*QRnnNetwork/EncodingNetwork/dense_1/MatMulMatMul4QRnnNetwork/EncodingNetwork/dense/Selu:activations:0AQRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЛ
:QRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpReadVariableOpCqrnnnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0у
+QRnnNetwork/EncodingNetwork/dense_1/BiasAddBiasAdd4QRnnNetwork/EncodingNetwork/dense_1/MatMul:product:0BQRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
(QRnnNetwork/EncodingNetwork/dense_1/SeluSelu4QRnnNetwork/EncodingNetwork/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
?QRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
AQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
AQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Љ
9QRnnNetwork/EncodingNetwork/batch_unflatten/strided_sliceStridedSlice8QRnnNetwork/EncodingNetwork/batch_flatten/Shape:output:0HQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack:output:0JQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_1:output:0JQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_maskЕ
1QRnnNetwork/EncodingNetwork/batch_unflatten/ShapeShape6QRnnNetwork/EncodingNetwork/dense_1/Selu:activations:0*
T0*
_output_shapes
:*
out_type0	:эа
AQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
CQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
CQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Б
;QRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1StridedSlice:QRnnNetwork/EncodingNetwork/batch_unflatten/Shape:output:0JQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack:output:0LQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_1:output:0LQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
end_masky
7QRnnNetwork/EncodingNetwork/batch_unflatten/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : И
2QRnnNetwork/EncodingNetwork/batch_unflatten/concatConcatV2BQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice:output:0DQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1:output:0@QRnnNetwork/EncodingNetwork/batch_unflatten/concat/axis:output:0*
N*
T0	*
_output_shapes
:ј
3QRnnNetwork/EncodingNetwork/batch_unflatten/ReshapeReshape6QRnnNetwork/EncodingNetwork/dense_1/Selu:activations:0;QRnnNetwork/EncodingNetwork/batch_unflatten/concat:output:0*
Tshape0	*
T0*,
_output_shapes
:џџџџџџџџџT
QRnnNetwork/mask/yConst*
_output_shapes
: *
dtype0*
value	B : 
QRnnNetwork/maskEqual!QRnnNetwork/ExpandDims_1:output:0QRnnNetwork/mask/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџa
QRnnNetwork/dynamic_unroll/RankConst*
_output_shapes
: *
dtype0*
value	B :h
&QRnnNetwork/dynamic_unroll/range/startConst*
_output_shapes
: *
dtype0*
value	B :h
&QRnnNetwork/dynamic_unroll/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :б
 QRnnNetwork/dynamic_unroll/rangeRange/QRnnNetwork/dynamic_unroll/range/start:output:0(QRnnNetwork/dynamic_unroll/Rank:output:0/QRnnNetwork/dynamic_unroll/range/delta:output:0*
_output_shapes
:{
*QRnnNetwork/dynamic_unroll/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB"       h
&QRnnNetwork/dynamic_unroll/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ь
!QRnnNetwork/dynamic_unroll/concatConcatV23QRnnNetwork/dynamic_unroll/concat/values_0:output:0)QRnnNetwork/dynamic_unroll/range:output:0/QRnnNetwork/dynamic_unroll/concat/axis:output:0*
N*
T0*
_output_shapes
:в
$QRnnNetwork/dynamic_unroll/transpose	Transpose<QRnnNetwork/EncodingNetwork/batch_unflatten/Reshape:output:0*QRnnNetwork/dynamic_unroll/concat:output:0*
T0*,
_output_shapes
:џџџџџџџџџ
 QRnnNetwork/dynamic_unroll/ShapeShape(QRnnNetwork/dynamic_unroll/transpose:y:0*
T0*
_output_shapes
::эЯx
.QRnnNetwork/dynamic_unroll/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:z
0QRnnNetwork/dynamic_unroll/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0QRnnNetwork/dynamic_unroll/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:и
(QRnnNetwork/dynamic_unroll/strided_sliceStridedSlice)QRnnNetwork/dynamic_unroll/Shape:output:07QRnnNetwork/dynamic_unroll/strided_slice/stack:output:09QRnnNetwork/dynamic_unroll/strided_slice/stack_1:output:09QRnnNetwork/dynamic_unroll/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask|
+QRnnNetwork/dynamic_unroll/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       Б
&QRnnNetwork/dynamic_unroll/transpose_1	TransposeQRnnNetwork/mask:z:04QRnnNetwork/dynamic_unroll/transpose_1/perm:output:0*
T0
*'
_output_shapes
:џџџџџџџџџk
)QRnnNetwork/dynamic_unroll/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :MФ
'QRnnNetwork/dynamic_unroll/zeros/packedPack1QRnnNetwork/dynamic_unroll/strided_slice:output:02QRnnNetwork/dynamic_unroll/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:k
&QRnnNetwork/dynamic_unroll/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Н
 QRnnNetwork/dynamic_unroll/zerosFill0QRnnNetwork/dynamic_unroll/zeros/packed:output:0/QRnnNetwork/dynamic_unroll/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџMm
+QRnnNetwork/dynamic_unroll/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :MШ
)QRnnNetwork/dynamic_unroll/zeros_1/packedPack1QRnnNetwork/dynamic_unroll/strided_slice:output:04QRnnNetwork/dynamic_unroll/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:m
(QRnnNetwork/dynamic_unroll/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    У
"QRnnNetwork/dynamic_unroll/zeros_1Fill2QRnnNetwork/dynamic_unroll/zeros_1/packed:output:01QRnnNetwork/dynamic_unroll/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџMЁ
"QRnnNetwork/dynamic_unroll/SqueezeSqueeze(QRnnNetwork/dynamic_unroll/transpose:y:0*
T0*(
_output_shapes
:џџџџџџџџџ*
squeeze_dims
  
$QRnnNetwork/dynamic_unroll/Squeeze_1Squeeze*QRnnNetwork/dynamic_unroll/transpose_1:y:0*
T0
*#
_output_shapes
:џџџџџџџџџ*
squeeze_dims
 г
!QRnnNetwork/dynamic_unroll/SelectSelect-QRnnNetwork/dynamic_unroll/Squeeze_1:output:0)QRnnNetwork/dynamic_unroll/zeros:output:0PartitionedCall_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџMз
#QRnnNetwork/dynamic_unroll/Select_1Select-QRnnNetwork/dynamic_unroll/Squeeze_1:output:0+QRnnNetwork/dynamic_unroll/zeros_1:output:0PartitionedCall_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџMР
:QRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOpReadVariableOpCqrnnnetwork_dynamic_unroll_lstm_cell_matmul_readvariableop_resource* 
_output_shapes
:
Д*
dtype0й
+QRnnNetwork/dynamic_unroll/lstm_cell/MatMulMatMul+QRnnNetwork/dynamic_unroll/Squeeze:output:0BQRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџДУ
<QRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpEqrnnnetwork_dynamic_unroll_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes
:	MД*
dtype0м
-QRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1MatMul*QRnnNetwork/dynamic_unroll/Select:output:0DQRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџДд
(QRnnNetwork/dynamic_unroll/lstm_cell/addAddV25QRnnNetwork/dynamic_unroll/lstm_cell/MatMul:product:07QRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџДН
;QRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOpReadVariableOpDqrnnnetwork_dynamic_unroll_lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:Д*
dtype0н
,QRnnNetwork/dynamic_unroll/lstm_cell/BiasAddBiasAdd,QRnnNetwork/dynamic_unroll/lstm_cell/add:z:0CQRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџДv
4QRnnNetwork/dynamic_unroll/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ѕ
*QRnnNetwork/dynamic_unroll/lstm_cell/splitSplit=QRnnNetwork/dynamic_unroll/lstm_cell/split/split_dim:output:05QRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџM:џџџџџџџџџM:џџџџџџџџџM:џџџџџџџџџM*
	num_split
,QRnnNetwork/dynamic_unroll/lstm_cell/SigmoidSigmoid3QRnnNetwork/dynamic_unroll/lstm_cell/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџM 
.QRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_1Sigmoid3QRnnNetwork/dynamic_unroll/lstm_cell/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџMУ
(QRnnNetwork/dynamic_unroll/lstm_cell/mulMul2QRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_1:y:0,QRnnNetwork/dynamic_unroll/Select_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџM
)QRnnNetwork/dynamic_unroll/lstm_cell/TanhTanh3QRnnNetwork/dynamic_unroll/lstm_cell/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџMФ
*QRnnNetwork/dynamic_unroll/lstm_cell/mul_1Mul0QRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid:y:0-QRnnNetwork/dynamic_unroll/lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџMУ
*QRnnNetwork/dynamic_unroll/lstm_cell/add_1AddV2,QRnnNetwork/dynamic_unroll/lstm_cell/mul:z:0.QRnnNetwork/dynamic_unroll/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџM 
.QRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_2Sigmoid3QRnnNetwork/dynamic_unroll/lstm_cell/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџM
+QRnnNetwork/dynamic_unroll/lstm_cell/Tanh_1Tanh.QRnnNetwork/dynamic_unroll/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџMШ
*QRnnNetwork/dynamic_unroll/lstm_cell/mul_2Mul2QRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_2:y:0/QRnnNetwork/dynamic_unroll/lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџMk
)QRnnNetwork/dynamic_unroll/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :Э
%QRnnNetwork/dynamic_unroll/ExpandDims
ExpandDims.QRnnNetwork/dynamic_unroll/lstm_cell/mul_2:z:02QRnnNetwork/dynamic_unroll/ExpandDims/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџMЃ
,QRnnNetwork/dense_2/Tensordot/ReadVariableOpReadVariableOp5qrnnnetwork_dense_2_tensordot_readvariableop_resource*
_output_shapes
:	Mи*
dtype0l
"QRnnNetwork/dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:s
"QRnnNetwork/dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       
#QRnnNetwork/dense_2/Tensordot/ShapeShape.QRnnNetwork/dynamic_unroll/ExpandDims:output:0*
T0*
_output_shapes
::эЯm
+QRnnNetwork/dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
&QRnnNetwork/dense_2/Tensordot/GatherV2GatherV2,QRnnNetwork/dense_2/Tensordot/Shape:output:0+QRnnNetwork/dense_2/Tensordot/free:output:04QRnnNetwork/dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:o
-QRnnNetwork/dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
(QRnnNetwork/dense_2/Tensordot/GatherV2_1GatherV2,QRnnNetwork/dense_2/Tensordot/Shape:output:0+QRnnNetwork/dense_2/Tensordot/axes:output:06QRnnNetwork/dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:m
#QRnnNetwork/dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: Њ
"QRnnNetwork/dense_2/Tensordot/ProdProd/QRnnNetwork/dense_2/Tensordot/GatherV2:output:0,QRnnNetwork/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: o
%QRnnNetwork/dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: А
$QRnnNetwork/dense_2/Tensordot/Prod_1Prod1QRnnNetwork/dense_2/Tensordot/GatherV2_1:output:0.QRnnNetwork/dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: k
)QRnnNetwork/dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ь
$QRnnNetwork/dense_2/Tensordot/concatConcatV2+QRnnNetwork/dense_2/Tensordot/free:output:0+QRnnNetwork/dense_2/Tensordot/axes:output:02QRnnNetwork/dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:Е
#QRnnNetwork/dense_2/Tensordot/stackPack+QRnnNetwork/dense_2/Tensordot/Prod:output:0-QRnnNetwork/dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Щ
'QRnnNetwork/dense_2/Tensordot/transpose	Transpose.QRnnNetwork/dynamic_unroll/ExpandDims:output:0-QRnnNetwork/dense_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџMЦ
%QRnnNetwork/dense_2/Tensordot/ReshapeReshape+QRnnNetwork/dense_2/Tensordot/transpose:y:0,QRnnNetwork/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџЧ
$QRnnNetwork/dense_2/Tensordot/MatMulMatMul.QRnnNetwork/dense_2/Tensordot/Reshape:output:04QRnnNetwork/dense_2/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџиp
%QRnnNetwork/dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:иm
+QRnnNetwork/dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ї
&QRnnNetwork/dense_2/Tensordot/concat_1ConcatV2/QRnnNetwork/dense_2/Tensordot/GatherV2:output:0.QRnnNetwork/dense_2/Tensordot/Const_2:output:04QRnnNetwork/dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Р
QRnnNetwork/dense_2/TensordotReshape.QRnnNetwork/dense_2/Tensordot/MatMul:product:0/QRnnNetwork/dense_2/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:џџџџџџџџџи
*QRnnNetwork/dense_2/BiasAdd/ReadVariableOpReadVariableOp3qrnnnetwork_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:и*
dtype0Й
QRnnNetwork/dense_2/BiasAddBiasAdd&QRnnNetwork/dense_2/Tensordot:output:02QRnnNetwork/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџи}
QRnnNetwork/dense_2/SeluSelu$QRnnNetwork/dense_2/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџиЃ
,QRnnNetwork/dense_3/Tensordot/ReadVariableOpReadVariableOp5qrnnnetwork_dense_3_tensordot_readvariableop_resource*
_output_shapes
:	иd*
dtype0l
"QRnnNetwork/dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:s
"QRnnNetwork/dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       
#QRnnNetwork/dense_3/Tensordot/ShapeShape&QRnnNetwork/dense_2/Selu:activations:0*
T0*
_output_shapes
::эЯm
+QRnnNetwork/dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
&QRnnNetwork/dense_3/Tensordot/GatherV2GatherV2,QRnnNetwork/dense_3/Tensordot/Shape:output:0+QRnnNetwork/dense_3/Tensordot/free:output:04QRnnNetwork/dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:o
-QRnnNetwork/dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
(QRnnNetwork/dense_3/Tensordot/GatherV2_1GatherV2,QRnnNetwork/dense_3/Tensordot/Shape:output:0+QRnnNetwork/dense_3/Tensordot/axes:output:06QRnnNetwork/dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:m
#QRnnNetwork/dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: Њ
"QRnnNetwork/dense_3/Tensordot/ProdProd/QRnnNetwork/dense_3/Tensordot/GatherV2:output:0,QRnnNetwork/dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: o
%QRnnNetwork/dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: А
$QRnnNetwork/dense_3/Tensordot/Prod_1Prod1QRnnNetwork/dense_3/Tensordot/GatherV2_1:output:0.QRnnNetwork/dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: k
)QRnnNetwork/dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ь
$QRnnNetwork/dense_3/Tensordot/concatConcatV2+QRnnNetwork/dense_3/Tensordot/free:output:0+QRnnNetwork/dense_3/Tensordot/axes:output:02QRnnNetwork/dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:Е
#QRnnNetwork/dense_3/Tensordot/stackPack+QRnnNetwork/dense_3/Tensordot/Prod:output:0-QRnnNetwork/dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Т
'QRnnNetwork/dense_3/Tensordot/transpose	Transpose&QRnnNetwork/dense_2/Selu:activations:0-QRnnNetwork/dense_3/Tensordot/concat:output:0*
T0*,
_output_shapes
:џџџџџџџџџиЦ
%QRnnNetwork/dense_3/Tensordot/ReshapeReshape+QRnnNetwork/dense_3/Tensordot/transpose:y:0,QRnnNetwork/dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџЦ
$QRnnNetwork/dense_3/Tensordot/MatMulMatMul.QRnnNetwork/dense_3/Tensordot/Reshape:output:04QRnnNetwork/dense_3/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџdo
%QRnnNetwork/dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:dm
+QRnnNetwork/dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ї
&QRnnNetwork/dense_3/Tensordot/concat_1ConcatV2/QRnnNetwork/dense_3/Tensordot/GatherV2:output:0.QRnnNetwork/dense_3/Tensordot/Const_2:output:04QRnnNetwork/dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:П
QRnnNetwork/dense_3/TensordotReshape.QRnnNetwork/dense_3/Tensordot/MatMul:product:0/QRnnNetwork/dense_3/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџd
*QRnnNetwork/dense_3/BiasAdd/ReadVariableOpReadVariableOp3qrnnnetwork_dense_3_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0И
QRnnNetwork/dense_3/BiasAddBiasAdd&QRnnNetwork/dense_3/Tensordot:output:02QRnnNetwork/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџd|
QRnnNetwork/dense_3/SeluSelu$QRnnNetwork/dense_3/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџdФ
=QRnnNetwork/num_action_project/dense/Tensordot/ReadVariableOpReadVariableOpFqrnnnetwork_num_action_project_dense_tensordot_readvariableop_resource*
_output_shapes

:d*
dtype0}
3QRnnNetwork/num_action_project/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
3QRnnNetwork/num_action_project/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       
4QRnnNetwork/num_action_project/dense/Tensordot/ShapeShape&QRnnNetwork/dense_3/Selu:activations:0*
T0*
_output_shapes
::эЯ~
<QRnnNetwork/num_action_project/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Я
7QRnnNetwork/num_action_project/dense/Tensordot/GatherV2GatherV2=QRnnNetwork/num_action_project/dense/Tensordot/Shape:output:0<QRnnNetwork/num_action_project/dense/Tensordot/free:output:0EQRnnNetwork/num_action_project/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
>QRnnNetwork/num_action_project/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : г
9QRnnNetwork/num_action_project/dense/Tensordot/GatherV2_1GatherV2=QRnnNetwork/num_action_project/dense/Tensordot/Shape:output:0<QRnnNetwork/num_action_project/dense/Tensordot/axes:output:0GQRnnNetwork/num_action_project/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:~
4QRnnNetwork/num_action_project/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: н
3QRnnNetwork/num_action_project/dense/Tensordot/ProdProd@QRnnNetwork/num_action_project/dense/Tensordot/GatherV2:output:0=QRnnNetwork/num_action_project/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 
6QRnnNetwork/num_action_project/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: у
5QRnnNetwork/num_action_project/dense/Tensordot/Prod_1ProdBQRnnNetwork/num_action_project/dense/Tensordot/GatherV2_1:output:0?QRnnNetwork/num_action_project/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: |
:QRnnNetwork/num_action_project/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : А
5QRnnNetwork/num_action_project/dense/Tensordot/concatConcatV2<QRnnNetwork/num_action_project/dense/Tensordot/free:output:0<QRnnNetwork/num_action_project/dense/Tensordot/axes:output:0CQRnnNetwork/num_action_project/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:ш
4QRnnNetwork/num_action_project/dense/Tensordot/stackPack<QRnnNetwork/num_action_project/dense/Tensordot/Prod:output:0>QRnnNetwork/num_action_project/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:у
8QRnnNetwork/num_action_project/dense/Tensordot/transpose	Transpose&QRnnNetwork/dense_3/Selu:activations:0>QRnnNetwork/num_action_project/dense/Tensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџdљ
6QRnnNetwork/num_action_project/dense/Tensordot/ReshapeReshape<QRnnNetwork/num_action_project/dense/Tensordot/transpose:y:0=QRnnNetwork/num_action_project/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџљ
5QRnnNetwork/num_action_project/dense/Tensordot/MatMulMatMul?QRnnNetwork/num_action_project/dense/Tensordot/Reshape:output:0EQRnnNetwork/num_action_project/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
6QRnnNetwork/num_action_project/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:~
<QRnnNetwork/num_action_project/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Л
7QRnnNetwork/num_action_project/dense/Tensordot/concat_1ConcatV2@QRnnNetwork/num_action_project/dense/Tensordot/GatherV2:output:0?QRnnNetwork/num_action_project/dense/Tensordot/Const_2:output:0EQRnnNetwork/num_action_project/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:ђ
.QRnnNetwork/num_action_project/dense/TensordotReshape?QRnnNetwork/num_action_project/dense/Tensordot/MatMul:product:0@QRnnNetwork/num_action_project/dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџМ
;QRnnNetwork/num_action_project/dense/BiasAdd/ReadVariableOpReadVariableOpDqrnnnetwork_num_action_project_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ы
,QRnnNetwork/num_action_project/dense/BiasAddBiasAdd7QRnnNetwork/num_action_project/dense/Tensordot:output:0CQRnnNetwork/num_action_project/dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ
QRnnNetwork/SqueezeSqueeze5QRnnNetwork/num_action_project/dense/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
squeeze_dims
l
!Categorical/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
Categorical/mode/ArgMaxArgMaxQRnnNetwork/Squeeze:output:0*Categorical/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:џџџџџџџџџ|
Categorical/mode/CastCast Categorical/mode/ArgMax:output:0*

DstT0*

SrcT0	*#
_output_shapes
:џџџџџџџџџW
Deterministic/atolConst*
_output_shapes
: *
dtype0*
valueB
 *    W
Deterministic/rtolConst*
_output_shapes
: *
dtype0*
valueB
 *    d
!Deterministic/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB q
Deterministic/sample/ShapeShapeCategorical/mode/Cast:y:0*
T0*
_output_shapes
::эЯ\
Deterministic/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : r
(Deterministic/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*Deterministic/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*Deterministic/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:И
"Deterministic/sample/strided_sliceStridedSlice#Deterministic/sample/Shape:output:01Deterministic/sample/strided_slice/stack:output:03Deterministic/sample/strided_slice/stack_1:output:03Deterministic/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskh
%Deterministic/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB j
'Deterministic/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB Ў
"Deterministic/sample/BroadcastArgsBroadcastArgs0Deterministic/sample/BroadcastArgs/s0_1:output:0+Deterministic/sample/strided_slice:output:0*
_output_shapes
:n
$Deterministic/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:g
$Deterministic/sample/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB b
 Deterministic/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Deterministic/sample/concatConcatV2-Deterministic/sample/concat/values_0:output:0'Deterministic/sample/BroadcastArgs:r0:0-Deterministic/sample/concat/values_2:output:0)Deterministic/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:Ђ
 Deterministic/sample/BroadcastToBroadcastToCategorical/mode/Cast:y:0$Deterministic/sample/concat:output:0*
T0*'
_output_shapes
:џџџџџџџџџe
"Deterministic/sample/Shape_1/ConstConst*
_output_shapes
: *
dtype0*
valueB f
Deterministic/sample/Shape_1Const*
_output_shapes
:*
dtype0*
valueB: 
Deterministic/sample/Shape_2Shape)Deterministic/sample/BroadcastTo:output:0*
T0*
_output_shapes
::эЯt
*Deterministic/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:v
,Deterministic/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,Deterministic/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Р
$Deterministic/sample/strided_slice_1StridedSlice%Deterministic/sample/Shape_2:output:03Deterministic/sample/strided_slice_1/stack:output:05Deterministic/sample/strided_slice_1/stack_1:output:05Deterministic/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskd
"Deterministic/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : п
Deterministic/sample/concat_1ConcatV2*Deterministic/sample/sample_shape:output:0-Deterministic/sample/strided_slice_1:output:0+Deterministic/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Ј
Deterministic/sample/ReshapeReshape)Deterministic/sample/BroadcastTo:output:0&Deterministic/sample/concat_1:output:0*
T0*#
_output_shapes
:џџџџџџџџџ\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @
clip_by_value/MinimumMinimum%Deterministic/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0*#
_output_shapes
:џџџџџџџџџT
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    {
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*#
_output_shapes
:џџџџџџџџџ\
IdentityIdentityclip_by_value:z:0^NoOp*
T0*#
_output_shapes
:џџџџџџџџџ

Identity_1Identity.QRnnNetwork/dynamic_unroll/lstm_cell/mul_2:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџM

Identity_2Identity.QRnnNetwork/dynamic_unroll/lstm_cell/add_1:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџM
NoOpNoOp9^QRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp8^QRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp;^QRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp:^QRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp+^QRnnNetwork/dense_2/BiasAdd/ReadVariableOp-^QRnnNetwork/dense_2/Tensordot/ReadVariableOp+^QRnnNetwork/dense_3/BiasAdd/ReadVariableOp-^QRnnNetwork/dense_3/Tensordot/ReadVariableOp<^QRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOp;^QRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOp=^QRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOp<^QRnnNetwork/num_action_project/dense/BiasAdd/ReadVariableOp>^QRnnNetwork/num_action_project/dense/Tensordot/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџM:џџџџџџџџџM: : : : : : : : : : : : : 2t
8QRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp8QRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp2r
7QRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp7QRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp2x
:QRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp:QRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp2v
9QRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp9QRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp2X
*QRnnNetwork/dense_2/BiasAdd/ReadVariableOp*QRnnNetwork/dense_2/BiasAdd/ReadVariableOp2\
,QRnnNetwork/dense_2/Tensordot/ReadVariableOp,QRnnNetwork/dense_2/Tensordot/ReadVariableOp2X
*QRnnNetwork/dense_3/BiasAdd/ReadVariableOp*QRnnNetwork/dense_3/BiasAdd/ReadVariableOp2\
,QRnnNetwork/dense_3/Tensordot/ReadVariableOp,QRnnNetwork/dense_3/Tensordot/ReadVariableOp2z
;QRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOp;QRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOp2x
:QRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOp:QRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOp2|
<QRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOp<QRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOp2z
;QRnnNetwork/num_action_project/dense/BiasAdd/ReadVariableOp;QRnnNetwork/num_action_project/dense/BiasAdd/ReadVariableOp2~
=QRnnNetwork/num_action_project/dense/Tensordot/ReadVariableOp=QRnnNetwork/num_action_project/dense/Tensordot/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:JF
'
_output_shapes
:џџџџџџџџџM

_user_specified_name1:JF
'
_output_shapes
:џџџџџџџџџM

_user_specified_name0:TP
'
_output_shapes
:џџџџџџџџџ
%
_user_specified_nameobservation:MI
#
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
discount:KG
#
_output_shapes
:џџџџџџџџџ
 
_user_specified_namereward:N J
#
_output_shapes
:џџџџџџџџџ
#
_user_specified_name	step_type
ж
,
*__inference_function_with_signature_294269ш
PartitionedCallPartitionedCall*	
Tin
 *

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *$
fR
__inference_<lambda>_293281*(
_construction_contextkEagerRuntime*
_input_shapes 
ї
>
<__inference_signature_wrapper_function_with_signature_294272ї
PartitionedCallPartitionedCall*	
Tin
 *

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *3
f.R,
*__inference_function_with_signature_294269*(
_construction_contextkEagerRuntime*
_input_shapes 
и
c
"__inference_per_field_where_294949
t
f

sub_rank_1
shape_equal_1

identityY
assert_rank_at_least/ShapeShapet*
T0*
_output_shapes
::эЯa
Cassert_rank_at_least/assert_type/statically_determined_correct_typeNoOp*
_output_shapes
 R
4assert_rank_at_least/static_checks_determined_all_okNoOp*
_output_shapes
 F
RankConst*
_output_shapes
: *
dtype0*
value	B :F
subSubRank:output:0
sub_rank_1*
T0*
_output_shapes
: P
ShapeShapeshape_equal_1*
T0
*
_output_shapes
::эЯe
ones/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџb
ones/ReshapeReshapesub:z:0ones/Reshape/shape:output:0*
T0*
_output_shapes
:L

ones/ConstConst*
_output_shapes
: *
dtype0*
value	B :f
onesFillones/Reshape:output:0ones/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ~
concatConcatV2Shape:output:0ones:output:0concat/axis:output:0*
N*
T0*#
_output_shapes
:џџџџџџџџџU
ReshapeReshapeshape_equal_1concat:output:0*
T0
*
_output_shapes
:O
SelectV2SelectV2Reshape:output:0tf*
T0*
_output_shapes
:J
IdentityIdentitySelectV2:output:0*
T0*
_output_shapes
:"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:џџџџџџџџџM:џџџџџџџџџM: :џџџџџџџџџ:LH
#
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	Equal_1:>:

_output_shapes
: 
 
_user_specified_nameRank_1:JF
'
_output_shapes
:џџџџџџџџџM

_user_specified_namef:J F
'
_output_shapes
:џџџџџџџџџM

_user_specified_namet
Ћд
и
__inference_action_294864
time_step_step_type
time_step_reward
time_step_discount
time_step_observation
policy_state_0
policy_state_1S
@qrnnnetwork_encodingnetwork_dense_matmul_readvariableop_resource:	бP
Aqrnnnetwork_encodingnetwork_dense_biasadd_readvariableop_resource:	бV
Bqrnnnetwork_encodingnetwork_dense_1_matmul_readvariableop_resource:
бR
Cqrnnnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resource:	W
Cqrnnnetwork_dynamic_unroll_lstm_cell_matmul_readvariableop_resource:
ДX
Eqrnnnetwork_dynamic_unroll_lstm_cell_matmul_1_readvariableop_resource:	MДS
Dqrnnnetwork_dynamic_unroll_lstm_cell_biasadd_readvariableop_resource:	ДH
5qrnnnetwork_dense_2_tensordot_readvariableop_resource:	MиB
3qrnnnetwork_dense_2_biasadd_readvariableop_resource:	иH
5qrnnnetwork_dense_3_tensordot_readvariableop_resource:	иdA
3qrnnnetwork_dense_3_biasadd_readvariableop_resource:dX
Fqrnnnetwork_num_action_project_dense_tensordot_readvariableop_resource:dR
Dqrnnnetwork_num_action_project_dense_biasadd_readvariableop_resource:
identity

identity_1

identity_2Ђ8QRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpЂ7QRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpЂ:QRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpЂ9QRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpЂ*QRnnNetwork/dense_2/BiasAdd/ReadVariableOpЂ,QRnnNetwork/dense_2/Tensordot/ReadVariableOpЂ*QRnnNetwork/dense_3/BiasAdd/ReadVariableOpЂ,QRnnNetwork/dense_3/Tensordot/ReadVariableOpЂ;QRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOpЂ:QRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOpЂ<QRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOpЂ;QRnnNetwork/num_action_project/dense/BiasAdd/ReadVariableOpЂ=QRnnNetwork/num_action_project/dense/Tensordot/ReadVariableOpU
ShapeShapetime_step_discount*
T0*
_output_shapes
::эЯ]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:б
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskT
packedPackstrided_slice:output:0*
N*
T0*
_output_shapes
:Y
shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:MM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
concatConcatV2packed:output:0shape_as_tensor:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    f
zerosFillconcat:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџM[
shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB:MO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
concat_1ConcatV2packed:output:0shape_as_tensor_1:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zeros_1Fillconcat_1:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџMI
Equal/yConst*
_output_shapes
: *
dtype0*
value	B : c
EqualEqualtime_step_step_typeEqual/y:output:0*
T0*#
_output_shapes
:џџџџџџџџџF
RankConst*
_output_shapes
: *
dtype0*
value	B :Х
PartitionedCallPartitionedCallzeros:output:0policy_state_0Rank:output:0	Equal:z:0*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџM* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference_per_field_where_294613Щ
PartitionedCall_1PartitionedCallzeros_1:output:0policy_state_1Rank:output:0	Equal:z:0*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџM* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference_per_field_where_294613W
Shape_1Shapetime_step_discount*
T0*
_output_shapes
::эЯ_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:л
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
packed_1Packstrided_slice_1:output:0*
N*
T0*
_output_shapes
:[
shape_as_tensor_2Const*
_output_shapes
:*
dtype0*
valueB:MO
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
concat_2ConcatV2packed_1:output:0shape_as_tensor_2:output:0concat_2/axis:output:0*
N*
T0*
_output_shapes
:R
zeros_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zeros_2Fillconcat_2:output:0zeros_2/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџM[
shape_as_tensor_3Const*
_output_shapes
:*
dtype0*
valueB:MO
concat_3/axisConst*
_output_shapes
: *
dtype0*
value	B : 
concat_3ConcatV2packed_1:output:0shape_as_tensor_3:output:0concat_3/axis:output:0*
N*
T0*
_output_shapes
:R
zeros_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zeros_3Fillconcat_3:output:0zeros_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџMK
	Equal_1/yConst*
_output_shapes
: *
dtype0*
value	B : g
Equal_1Equaltime_step_step_typeEqual_1/y:output:0*
T0*#
_output_shapes
:џџџџџџџџџH
Rank_1Const*
_output_shapes
: *
dtype0*
value	B :з
PartitionedCall_2PartitionedCallzeros_2:output:0PartitionedCall:output:0Rank_1:output:0Equal_1:z:0*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџM* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference_per_field_where_294653й
PartitionedCall_3PartitionedCallzeros_3:output:0PartitionedCall_1:output:0Rank_1:output:0Equal_1:z:0*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџM* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference_per_field_where_294653\
QRnnNetwork/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :
QRnnNetwork/ExpandDims
ExpandDimstime_step_observation#QRnnNetwork/ExpandDims/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџ^
QRnnNetwork/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :
QRnnNetwork/ExpandDims_1
ExpandDimstime_step_step_type%QRnnNetwork/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
/QRnnNetwork/EncodingNetwork/batch_flatten/ShapeShapeQRnnNetwork/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	:эа
7QRnnNetwork/EncodingNetwork/batch_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   б
1QRnnNetwork/EncodingNetwork/batch_flatten/ReshapeReshapeQRnnNetwork/ExpandDims:output:0@QRnnNetwork/EncodingNetwork/batch_flatten/Reshape/shape:output:0*
T0*'
_output_shapes
:џџџџџџџџџz
)QRnnNetwork/EncodingNetwork/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   и
+QRnnNetwork/EncodingNetwork/flatten/ReshapeReshape:QRnnNetwork/EncodingNetwork/batch_flatten/Reshape:output:02QRnnNetwork/EncodingNetwork/flatten/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџЙ
7QRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpReadVariableOp@qrnnnetwork_encodingnetwork_dense_matmul_readvariableop_resource*
_output_shapes
:	б*
dtype0м
(QRnnNetwork/EncodingNetwork/dense/MatMulMatMul4QRnnNetwork/EncodingNetwork/flatten/Reshape:output:0?QRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџбЗ
8QRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpReadVariableOpAqrnnnetwork_encodingnetwork_dense_biasadd_readvariableop_resource*
_output_shapes	
:б*
dtype0н
)QRnnNetwork/EncodingNetwork/dense/BiasAddBiasAdd2QRnnNetwork/EncodingNetwork/dense/MatMul:product:0@QRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџб
&QRnnNetwork/EncodingNetwork/dense/SeluSelu2QRnnNetwork/EncodingNetwork/dense/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџбО
9QRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOpReadVariableOpBqrnnnetwork_encodingnetwork_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
б*
dtype0р
*QRnnNetwork/EncodingNetwork/dense_1/MatMulMatMul4QRnnNetwork/EncodingNetwork/dense/Selu:activations:0AQRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЛ
:QRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOpReadVariableOpCqrnnnetwork_encodingnetwork_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0у
+QRnnNetwork/EncodingNetwork/dense_1/BiasAddBiasAdd4QRnnNetwork/EncodingNetwork/dense_1/MatMul:product:0BQRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ
(QRnnNetwork/EncodingNetwork/dense_1/SeluSelu4QRnnNetwork/EncodingNetwork/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
?QRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
AQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
AQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Љ
9QRnnNetwork/EncodingNetwork/batch_unflatten/strided_sliceStridedSlice8QRnnNetwork/EncodingNetwork/batch_flatten/Shape:output:0HQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack:output:0JQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_1:output:0JQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_maskЕ
1QRnnNetwork/EncodingNetwork/batch_unflatten/ShapeShape6QRnnNetwork/EncodingNetwork/dense_1/Selu:activations:0*
T0*
_output_shapes
:*
out_type0	:эа
AQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
CQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
CQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Б
;QRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1StridedSlice:QRnnNetwork/EncodingNetwork/batch_unflatten/Shape:output:0JQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack:output:0LQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_1:output:0LQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
end_masky
7QRnnNetwork/EncodingNetwork/batch_unflatten/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : И
2QRnnNetwork/EncodingNetwork/batch_unflatten/concatConcatV2BQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice:output:0DQRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1:output:0@QRnnNetwork/EncodingNetwork/batch_unflatten/concat/axis:output:0*
N*
T0	*
_output_shapes
:ј
3QRnnNetwork/EncodingNetwork/batch_unflatten/ReshapeReshape6QRnnNetwork/EncodingNetwork/dense_1/Selu:activations:0;QRnnNetwork/EncodingNetwork/batch_unflatten/concat:output:0*
Tshape0	*
T0*,
_output_shapes
:џџџџџџџџџT
QRnnNetwork/mask/yConst*
_output_shapes
: *
dtype0*
value	B : 
QRnnNetwork/maskEqual!QRnnNetwork/ExpandDims_1:output:0QRnnNetwork/mask/y:output:0*
T0*'
_output_shapes
:џџџџџџџџџa
QRnnNetwork/dynamic_unroll/RankConst*
_output_shapes
: *
dtype0*
value	B :h
&QRnnNetwork/dynamic_unroll/range/startConst*
_output_shapes
: *
dtype0*
value	B :h
&QRnnNetwork/dynamic_unroll/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :б
 QRnnNetwork/dynamic_unroll/rangeRange/QRnnNetwork/dynamic_unroll/range/start:output:0(QRnnNetwork/dynamic_unroll/Rank:output:0/QRnnNetwork/dynamic_unroll/range/delta:output:0*
_output_shapes
:{
*QRnnNetwork/dynamic_unroll/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB"       h
&QRnnNetwork/dynamic_unroll/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ь
!QRnnNetwork/dynamic_unroll/concatConcatV23QRnnNetwork/dynamic_unroll/concat/values_0:output:0)QRnnNetwork/dynamic_unroll/range:output:0/QRnnNetwork/dynamic_unroll/concat/axis:output:0*
N*
T0*
_output_shapes
:в
$QRnnNetwork/dynamic_unroll/transpose	Transpose<QRnnNetwork/EncodingNetwork/batch_unflatten/Reshape:output:0*QRnnNetwork/dynamic_unroll/concat:output:0*
T0*,
_output_shapes
:џџџџџџџџџ
 QRnnNetwork/dynamic_unroll/ShapeShape(QRnnNetwork/dynamic_unroll/transpose:y:0*
T0*
_output_shapes
::эЯx
.QRnnNetwork/dynamic_unroll/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:z
0QRnnNetwork/dynamic_unroll/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0QRnnNetwork/dynamic_unroll/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:и
(QRnnNetwork/dynamic_unroll/strided_sliceStridedSlice)QRnnNetwork/dynamic_unroll/Shape:output:07QRnnNetwork/dynamic_unroll/strided_slice/stack:output:09QRnnNetwork/dynamic_unroll/strided_slice/stack_1:output:09QRnnNetwork/dynamic_unroll/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask|
+QRnnNetwork/dynamic_unroll/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       Б
&QRnnNetwork/dynamic_unroll/transpose_1	TransposeQRnnNetwork/mask:z:04QRnnNetwork/dynamic_unroll/transpose_1/perm:output:0*
T0
*'
_output_shapes
:џџџџџџџџџk
)QRnnNetwork/dynamic_unroll/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :MФ
'QRnnNetwork/dynamic_unroll/zeros/packedPack1QRnnNetwork/dynamic_unroll/strided_slice:output:02QRnnNetwork/dynamic_unroll/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:k
&QRnnNetwork/dynamic_unroll/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    Н
 QRnnNetwork/dynamic_unroll/zerosFill0QRnnNetwork/dynamic_unroll/zeros/packed:output:0/QRnnNetwork/dynamic_unroll/zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџMm
+QRnnNetwork/dynamic_unroll/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :MШ
)QRnnNetwork/dynamic_unroll/zeros_1/packedPack1QRnnNetwork/dynamic_unroll/strided_slice:output:04QRnnNetwork/dynamic_unroll/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:m
(QRnnNetwork/dynamic_unroll/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    У
"QRnnNetwork/dynamic_unroll/zeros_1Fill2QRnnNetwork/dynamic_unroll/zeros_1/packed:output:01QRnnNetwork/dynamic_unroll/zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџMЁ
"QRnnNetwork/dynamic_unroll/SqueezeSqueeze(QRnnNetwork/dynamic_unroll/transpose:y:0*
T0*(
_output_shapes
:џџџџџџџџџ*
squeeze_dims
  
$QRnnNetwork/dynamic_unroll/Squeeze_1Squeeze*QRnnNetwork/dynamic_unroll/transpose_1:y:0*
T0
*#
_output_shapes
:џџџџџџџџџ*
squeeze_dims
 г
!QRnnNetwork/dynamic_unroll/SelectSelect-QRnnNetwork/dynamic_unroll/Squeeze_1:output:0)QRnnNetwork/dynamic_unroll/zeros:output:0PartitionedCall_2:output:0*
T0*'
_output_shapes
:џџџџџџџџџMз
#QRnnNetwork/dynamic_unroll/Select_1Select-QRnnNetwork/dynamic_unroll/Squeeze_1:output:0+QRnnNetwork/dynamic_unroll/zeros_1:output:0PartitionedCall_3:output:0*
T0*'
_output_shapes
:џџџџџџџџџMР
:QRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOpReadVariableOpCqrnnnetwork_dynamic_unroll_lstm_cell_matmul_readvariableop_resource* 
_output_shapes
:
Д*
dtype0й
+QRnnNetwork/dynamic_unroll/lstm_cell/MatMulMatMul+QRnnNetwork/dynamic_unroll/Squeeze:output:0BQRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџДУ
<QRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpEqrnnnetwork_dynamic_unroll_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes
:	MД*
dtype0м
-QRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1MatMul*QRnnNetwork/dynamic_unroll/Select:output:0DQRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџДд
(QRnnNetwork/dynamic_unroll/lstm_cell/addAddV25QRnnNetwork/dynamic_unroll/lstm_cell/MatMul:product:07QRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:џџџџџџџџџДН
;QRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOpReadVariableOpDqrnnnetwork_dynamic_unroll_lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:Д*
dtype0н
,QRnnNetwork/dynamic_unroll/lstm_cell/BiasAddBiasAdd,QRnnNetwork/dynamic_unroll/lstm_cell/add:z:0CQRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџДv
4QRnnNetwork/dynamic_unroll/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ѕ
*QRnnNetwork/dynamic_unroll/lstm_cell/splitSplit=QRnnNetwork/dynamic_unroll/lstm_cell/split/split_dim:output:05QRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:џџџџџџџџџM:џџџџџџџџџM:џџџџџџџџџM:џџџџџџџџџM*
	num_split
,QRnnNetwork/dynamic_unroll/lstm_cell/SigmoidSigmoid3QRnnNetwork/dynamic_unroll/lstm_cell/split:output:0*
T0*'
_output_shapes
:џџџџџџџџџM 
.QRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_1Sigmoid3QRnnNetwork/dynamic_unroll/lstm_cell/split:output:1*
T0*'
_output_shapes
:џџџџџџџџџMУ
(QRnnNetwork/dynamic_unroll/lstm_cell/mulMul2QRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_1:y:0,QRnnNetwork/dynamic_unroll/Select_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџM
)QRnnNetwork/dynamic_unroll/lstm_cell/TanhTanh3QRnnNetwork/dynamic_unroll/lstm_cell/split:output:2*
T0*'
_output_shapes
:џџџџџџџџџMФ
*QRnnNetwork/dynamic_unroll/lstm_cell/mul_1Mul0QRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid:y:0-QRnnNetwork/dynamic_unroll/lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:џџџџџџџџџMУ
*QRnnNetwork/dynamic_unroll/lstm_cell/add_1AddV2,QRnnNetwork/dynamic_unroll/lstm_cell/mul:z:0.QRnnNetwork/dynamic_unroll/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџM 
.QRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_2Sigmoid3QRnnNetwork/dynamic_unroll/lstm_cell/split:output:3*
T0*'
_output_shapes
:џџџџџџџџџM
+QRnnNetwork/dynamic_unroll/lstm_cell/Tanh_1Tanh.QRnnNetwork/dynamic_unroll/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџMШ
*QRnnNetwork/dynamic_unroll/lstm_cell/mul_2Mul2QRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_2:y:0/QRnnNetwork/dynamic_unroll/lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџMk
)QRnnNetwork/dynamic_unroll/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :Э
%QRnnNetwork/dynamic_unroll/ExpandDims
ExpandDims.QRnnNetwork/dynamic_unroll/lstm_cell/mul_2:z:02QRnnNetwork/dynamic_unroll/ExpandDims/dim:output:0*
T0*+
_output_shapes
:џџџџџџџџџMЃ
,QRnnNetwork/dense_2/Tensordot/ReadVariableOpReadVariableOp5qrnnnetwork_dense_2_tensordot_readvariableop_resource*
_output_shapes
:	Mи*
dtype0l
"QRnnNetwork/dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:s
"QRnnNetwork/dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       
#QRnnNetwork/dense_2/Tensordot/ShapeShape.QRnnNetwork/dynamic_unroll/ExpandDims:output:0*
T0*
_output_shapes
::эЯm
+QRnnNetwork/dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
&QRnnNetwork/dense_2/Tensordot/GatherV2GatherV2,QRnnNetwork/dense_2/Tensordot/Shape:output:0+QRnnNetwork/dense_2/Tensordot/free:output:04QRnnNetwork/dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:o
-QRnnNetwork/dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
(QRnnNetwork/dense_2/Tensordot/GatherV2_1GatherV2,QRnnNetwork/dense_2/Tensordot/Shape:output:0+QRnnNetwork/dense_2/Tensordot/axes:output:06QRnnNetwork/dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:m
#QRnnNetwork/dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: Њ
"QRnnNetwork/dense_2/Tensordot/ProdProd/QRnnNetwork/dense_2/Tensordot/GatherV2:output:0,QRnnNetwork/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: o
%QRnnNetwork/dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: А
$QRnnNetwork/dense_2/Tensordot/Prod_1Prod1QRnnNetwork/dense_2/Tensordot/GatherV2_1:output:0.QRnnNetwork/dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: k
)QRnnNetwork/dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ь
$QRnnNetwork/dense_2/Tensordot/concatConcatV2+QRnnNetwork/dense_2/Tensordot/free:output:0+QRnnNetwork/dense_2/Tensordot/axes:output:02QRnnNetwork/dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:Е
#QRnnNetwork/dense_2/Tensordot/stackPack+QRnnNetwork/dense_2/Tensordot/Prod:output:0-QRnnNetwork/dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Щ
'QRnnNetwork/dense_2/Tensordot/transpose	Transpose.QRnnNetwork/dynamic_unroll/ExpandDims:output:0-QRnnNetwork/dense_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџMЦ
%QRnnNetwork/dense_2/Tensordot/ReshapeReshape+QRnnNetwork/dense_2/Tensordot/transpose:y:0,QRnnNetwork/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџЧ
$QRnnNetwork/dense_2/Tensordot/MatMulMatMul.QRnnNetwork/dense_2/Tensordot/Reshape:output:04QRnnNetwork/dense_2/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџиp
%QRnnNetwork/dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:иm
+QRnnNetwork/dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ї
&QRnnNetwork/dense_2/Tensordot/concat_1ConcatV2/QRnnNetwork/dense_2/Tensordot/GatherV2:output:0.QRnnNetwork/dense_2/Tensordot/Const_2:output:04QRnnNetwork/dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Р
QRnnNetwork/dense_2/TensordotReshape.QRnnNetwork/dense_2/Tensordot/MatMul:product:0/QRnnNetwork/dense_2/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:џџџџџџџџџи
*QRnnNetwork/dense_2/BiasAdd/ReadVariableOpReadVariableOp3qrnnnetwork_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:и*
dtype0Й
QRnnNetwork/dense_2/BiasAddBiasAdd&QRnnNetwork/dense_2/Tensordot:output:02QRnnNetwork/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:џџџџџџџџџи}
QRnnNetwork/dense_2/SeluSelu$QRnnNetwork/dense_2/BiasAdd:output:0*
T0*,
_output_shapes
:џџџџџџџџџиЃ
,QRnnNetwork/dense_3/Tensordot/ReadVariableOpReadVariableOp5qrnnnetwork_dense_3_tensordot_readvariableop_resource*
_output_shapes
:	иd*
dtype0l
"QRnnNetwork/dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:s
"QRnnNetwork/dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       
#QRnnNetwork/dense_3/Tensordot/ShapeShape&QRnnNetwork/dense_2/Selu:activations:0*
T0*
_output_shapes
::эЯm
+QRnnNetwork/dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
&QRnnNetwork/dense_3/Tensordot/GatherV2GatherV2,QRnnNetwork/dense_3/Tensordot/Shape:output:0+QRnnNetwork/dense_3/Tensordot/free:output:04QRnnNetwork/dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:o
-QRnnNetwork/dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
(QRnnNetwork/dense_3/Tensordot/GatherV2_1GatherV2,QRnnNetwork/dense_3/Tensordot/Shape:output:0+QRnnNetwork/dense_3/Tensordot/axes:output:06QRnnNetwork/dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:m
#QRnnNetwork/dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: Њ
"QRnnNetwork/dense_3/Tensordot/ProdProd/QRnnNetwork/dense_3/Tensordot/GatherV2:output:0,QRnnNetwork/dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: o
%QRnnNetwork/dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: А
$QRnnNetwork/dense_3/Tensordot/Prod_1Prod1QRnnNetwork/dense_3/Tensordot/GatherV2_1:output:0.QRnnNetwork/dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: k
)QRnnNetwork/dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ь
$QRnnNetwork/dense_3/Tensordot/concatConcatV2+QRnnNetwork/dense_3/Tensordot/free:output:0+QRnnNetwork/dense_3/Tensordot/axes:output:02QRnnNetwork/dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:Е
#QRnnNetwork/dense_3/Tensordot/stackPack+QRnnNetwork/dense_3/Tensordot/Prod:output:0-QRnnNetwork/dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Т
'QRnnNetwork/dense_3/Tensordot/transpose	Transpose&QRnnNetwork/dense_2/Selu:activations:0-QRnnNetwork/dense_3/Tensordot/concat:output:0*
T0*,
_output_shapes
:џџџџџџџџџиЦ
%QRnnNetwork/dense_3/Tensordot/ReshapeReshape+QRnnNetwork/dense_3/Tensordot/transpose:y:0,QRnnNetwork/dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџЦ
$QRnnNetwork/dense_3/Tensordot/MatMulMatMul.QRnnNetwork/dense_3/Tensordot/Reshape:output:04QRnnNetwork/dense_3/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџdo
%QRnnNetwork/dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:dm
+QRnnNetwork/dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ї
&QRnnNetwork/dense_3/Tensordot/concat_1ConcatV2/QRnnNetwork/dense_3/Tensordot/GatherV2:output:0.QRnnNetwork/dense_3/Tensordot/Const_2:output:04QRnnNetwork/dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:П
QRnnNetwork/dense_3/TensordotReshape.QRnnNetwork/dense_3/Tensordot/MatMul:product:0/QRnnNetwork/dense_3/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџd
*QRnnNetwork/dense_3/BiasAdd/ReadVariableOpReadVariableOp3qrnnnetwork_dense_3_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0И
QRnnNetwork/dense_3/BiasAddBiasAdd&QRnnNetwork/dense_3/Tensordot:output:02QRnnNetwork/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџd|
QRnnNetwork/dense_3/SeluSelu$QRnnNetwork/dense_3/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџdФ
=QRnnNetwork/num_action_project/dense/Tensordot/ReadVariableOpReadVariableOpFqrnnnetwork_num_action_project_dense_tensordot_readvariableop_resource*
_output_shapes

:d*
dtype0}
3QRnnNetwork/num_action_project/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
3QRnnNetwork/num_action_project/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       
4QRnnNetwork/num_action_project/dense/Tensordot/ShapeShape&QRnnNetwork/dense_3/Selu:activations:0*
T0*
_output_shapes
::эЯ~
<QRnnNetwork/num_action_project/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Я
7QRnnNetwork/num_action_project/dense/Tensordot/GatherV2GatherV2=QRnnNetwork/num_action_project/dense/Tensordot/Shape:output:0<QRnnNetwork/num_action_project/dense/Tensordot/free:output:0EQRnnNetwork/num_action_project/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
>QRnnNetwork/num_action_project/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : г
9QRnnNetwork/num_action_project/dense/Tensordot/GatherV2_1GatherV2=QRnnNetwork/num_action_project/dense/Tensordot/Shape:output:0<QRnnNetwork/num_action_project/dense/Tensordot/axes:output:0GQRnnNetwork/num_action_project/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:~
4QRnnNetwork/num_action_project/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: н
3QRnnNetwork/num_action_project/dense/Tensordot/ProdProd@QRnnNetwork/num_action_project/dense/Tensordot/GatherV2:output:0=QRnnNetwork/num_action_project/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 
6QRnnNetwork/num_action_project/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: у
5QRnnNetwork/num_action_project/dense/Tensordot/Prod_1ProdBQRnnNetwork/num_action_project/dense/Tensordot/GatherV2_1:output:0?QRnnNetwork/num_action_project/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: |
:QRnnNetwork/num_action_project/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : А
5QRnnNetwork/num_action_project/dense/Tensordot/concatConcatV2<QRnnNetwork/num_action_project/dense/Tensordot/free:output:0<QRnnNetwork/num_action_project/dense/Tensordot/axes:output:0CQRnnNetwork/num_action_project/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:ш
4QRnnNetwork/num_action_project/dense/Tensordot/stackPack<QRnnNetwork/num_action_project/dense/Tensordot/Prod:output:0>QRnnNetwork/num_action_project/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:у
8QRnnNetwork/num_action_project/dense/Tensordot/transpose	Transpose&QRnnNetwork/dense_3/Selu:activations:0>QRnnNetwork/num_action_project/dense/Tensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџdљ
6QRnnNetwork/num_action_project/dense/Tensordot/ReshapeReshape<QRnnNetwork/num_action_project/dense/Tensordot/transpose:y:0=QRnnNetwork/num_action_project/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџљ
5QRnnNetwork/num_action_project/dense/Tensordot/MatMulMatMul?QRnnNetwork/num_action_project/dense/Tensordot/Reshape:output:0EQRnnNetwork/num_action_project/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
6QRnnNetwork/num_action_project/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:~
<QRnnNetwork/num_action_project/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Л
7QRnnNetwork/num_action_project/dense/Tensordot/concat_1ConcatV2@QRnnNetwork/num_action_project/dense/Tensordot/GatherV2:output:0?QRnnNetwork/num_action_project/dense/Tensordot/Const_2:output:0EQRnnNetwork/num_action_project/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:ђ
.QRnnNetwork/num_action_project/dense/TensordotReshape?QRnnNetwork/num_action_project/dense/Tensordot/MatMul:product:0@QRnnNetwork/num_action_project/dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџМ
;QRnnNetwork/num_action_project/dense/BiasAdd/ReadVariableOpReadVariableOpDqrnnnetwork_num_action_project_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ы
,QRnnNetwork/num_action_project/dense/BiasAddBiasAdd7QRnnNetwork/num_action_project/dense/Tensordot:output:0CQRnnNetwork/num_action_project/dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ
QRnnNetwork/SqueezeSqueeze5QRnnNetwork/num_action_project/dense/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ*
squeeze_dims
l
!Categorical/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџ
Categorical/mode/ArgMaxArgMaxQRnnNetwork/Squeeze:output:0*Categorical/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:џџџџџџџџџ|
Categorical/mode/CastCast Categorical/mode/ArgMax:output:0*

DstT0*

SrcT0	*#
_output_shapes
:џџџџџџџџџW
Deterministic/atolConst*
_output_shapes
: *
dtype0*
valueB
 *    W
Deterministic/rtolConst*
_output_shapes
: *
dtype0*
valueB
 *    d
!Deterministic/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB q
Deterministic/sample/ShapeShapeCategorical/mode/Cast:y:0*
T0*
_output_shapes
::эЯ\
Deterministic/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : r
(Deterministic/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*Deterministic/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*Deterministic/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:И
"Deterministic/sample/strided_sliceStridedSlice#Deterministic/sample/Shape:output:01Deterministic/sample/strided_slice/stack:output:03Deterministic/sample/strided_slice/stack_1:output:03Deterministic/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskh
%Deterministic/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB j
'Deterministic/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB Ў
"Deterministic/sample/BroadcastArgsBroadcastArgs0Deterministic/sample/BroadcastArgs/s0_1:output:0+Deterministic/sample/strided_slice:output:0*
_output_shapes
:n
$Deterministic/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:g
$Deterministic/sample/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB b
 Deterministic/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Deterministic/sample/concatConcatV2-Deterministic/sample/concat/values_0:output:0'Deterministic/sample/BroadcastArgs:r0:0-Deterministic/sample/concat/values_2:output:0)Deterministic/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:Ђ
 Deterministic/sample/BroadcastToBroadcastToCategorical/mode/Cast:y:0$Deterministic/sample/concat:output:0*
T0*'
_output_shapes
:џџџџџџџџџe
"Deterministic/sample/Shape_1/ConstConst*
_output_shapes
: *
dtype0*
valueB f
Deterministic/sample/Shape_1Const*
_output_shapes
:*
dtype0*
valueB: 
Deterministic/sample/Shape_2Shape)Deterministic/sample/BroadcastTo:output:0*
T0*
_output_shapes
::эЯt
*Deterministic/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:v
,Deterministic/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: v
,Deterministic/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Р
$Deterministic/sample/strided_slice_1StridedSlice%Deterministic/sample/Shape_2:output:03Deterministic/sample/strided_slice_1/stack:output:05Deterministic/sample/strided_slice_1/stack_1:output:05Deterministic/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskd
"Deterministic/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : п
Deterministic/sample/concat_1ConcatV2*Deterministic/sample/sample_shape:output:0-Deterministic/sample/strided_slice_1:output:0+Deterministic/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Ј
Deterministic/sample/ReshapeReshape)Deterministic/sample/BroadcastTo:output:0&Deterministic/sample/concat_1:output:0*
T0*#
_output_shapes
:џџџџџџџџџ\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @
clip_by_value/MinimumMinimum%Deterministic/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0*#
_output_shapes
:џџџџџџџџџT
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    {
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*#
_output_shapes
:џџџџџџџџџ\
IdentityIdentityclip_by_value:z:0^NoOp*
T0*#
_output_shapes
:џџџџџџџџџ

Identity_1Identity.QRnnNetwork/dynamic_unroll/lstm_cell/mul_2:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџM

Identity_2Identity.QRnnNetwork/dynamic_unroll/lstm_cell/add_1:z:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџM
NoOpNoOp9^QRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp8^QRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp;^QRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp:^QRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp+^QRnnNetwork/dense_2/BiasAdd/ReadVariableOp-^QRnnNetwork/dense_2/Tensordot/ReadVariableOp+^QRnnNetwork/dense_3/BiasAdd/ReadVariableOp-^QRnnNetwork/dense_3/Tensordot/ReadVariableOp<^QRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOp;^QRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOp=^QRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOp<^QRnnNetwork/num_action_project/dense/BiasAdd/ReadVariableOp>^QRnnNetwork/num_action_project/dense/Tensordot/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџM:џџџџџџџџџM: : : : : : : : : : : : : 2t
8QRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp8QRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp2r
7QRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp7QRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp2x
:QRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp:QRnnNetwork/EncodingNetwork/dense_1/BiasAdd/ReadVariableOp2v
9QRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp9QRnnNetwork/EncodingNetwork/dense_1/MatMul/ReadVariableOp2X
*QRnnNetwork/dense_2/BiasAdd/ReadVariableOp*QRnnNetwork/dense_2/BiasAdd/ReadVariableOp2\
,QRnnNetwork/dense_2/Tensordot/ReadVariableOp,QRnnNetwork/dense_2/Tensordot/ReadVariableOp2X
*QRnnNetwork/dense_3/BiasAdd/ReadVariableOp*QRnnNetwork/dense_3/BiasAdd/ReadVariableOp2\
,QRnnNetwork/dense_3/Tensordot/ReadVariableOp,QRnnNetwork/dense_3/Tensordot/ReadVariableOp2z
;QRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOp;QRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOp2x
:QRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOp:QRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOp2|
<QRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOp<QRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOp2z
;QRnnNetwork/num_action_project/dense/BiasAdd/ReadVariableOp;QRnnNetwork/num_action_project/dense/BiasAdd/ReadVariableOp2~
=QRnnNetwork/num_action_project/dense/Tensordot/ReadVariableOp=QRnnNetwork/num_action_project/dense/Tensordot/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:WS
'
_output_shapes
:џџџџџџџџџM
(
_user_specified_namepolicy_state_1:WS
'
_output_shapes
:џџџџџџџџџM
(
_user_specified_namepolicy_state_0:^Z
'
_output_shapes
:џџџџџџџџџ
/
_user_specified_nametime_step_observation:WS
#
_output_shapes
:џџџџџџџџџ
,
_user_specified_nametime_step_discount:UQ
#
_output_shapes
:џџџџџџџџџ
*
_user_specified_nametime_step_reward:X T
#
_output_shapes
:џџџџџџџџџ
-
_user_specified_nametime_step_step_type
§
T
$__inference_get_initial_state_295149

batch_size
identity

identity_1H
packedPack
batch_size*
N*
T0*
_output_shapes
:Y
shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:MM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
concatConcatV2packed:output:0shape_as_tensor:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    f
zerosFillconcat:output:0zeros/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџM[
shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB:MO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
concat_1ConcatV2packed:output:0shape_as_tensor_1:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zeros_1Fillconcat_1:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџMV
IdentityIdentityzeros:output:0*
T0*'
_output_shapes
:џџџџџџџџџMZ

Identity_1Identityzeros_1:output:0*
T0*'
_output_shapes
:џџџџџџџџџM"!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
д{
ъ
__inference__traced_save_295266
file_prefix)
read_disablecopyonread_variable:	 T
Aread_1_disablecopyonread_qrnnnetwork_encodingnetwork_dense_kernel:	бN
?read_2_disablecopyonread_qrnnnetwork_encodingnetwork_dense_bias:	бW
Cread_3_disablecopyonread_qrnnnetwork_encodingnetwork_dense_1_kernel:
бP
Aread_4_disablecopyonread_qrnnnetwork_encodingnetwork_dense_1_bias:	N
:read_5_disablecopyonread_qrnnnetwork_dynamic_unroll_kernel:
ДW
Dread_6_disablecopyonread_qrnnnetwork_dynamic_unroll_recurrent_kernel:	MДG
8read_7_disablecopyonread_qrnnnetwork_dynamic_unroll_bias:	ДF
3read_8_disablecopyonread_qrnnnetwork_dense_2_kernel:	Mи@
1read_9_disablecopyonread_qrnnnetwork_dense_2_bias:	иG
4read_10_disablecopyonread_qrnnnetwork_dense_3_kernel:	иd@
2read_11_disablecopyonread_qrnnnetwork_dense_3_bias:dW
Eread_12_disablecopyonread_qrnnnetwork_num_action_project_dense_kernel:dQ
Cread_13_disablecopyonread_qrnnnetwork_num_action_project_dense_bias:
savev2_const
identity_29ЂMergeV2CheckpointsЂRead/DisableCopyOnReadЂRead/ReadVariableOpЂRead_1/DisableCopyOnReadЂRead_1/ReadVariableOpЂRead_10/DisableCopyOnReadЂRead_10/ReadVariableOpЂRead_11/DisableCopyOnReadЂRead_11/ReadVariableOpЂRead_12/DisableCopyOnReadЂRead_12/ReadVariableOpЂRead_13/DisableCopyOnReadЂRead_13/ReadVariableOpЂRead_2/DisableCopyOnReadЂRead_2/ReadVariableOpЂRead_3/DisableCopyOnReadЂRead_3/ReadVariableOpЂRead_4/DisableCopyOnReadЂRead_4/ReadVariableOpЂRead_5/DisableCopyOnReadЂRead_5/ReadVariableOpЂRead_6/DisableCopyOnReadЂRead_6/ReadVariableOpЂRead_7/DisableCopyOnReadЂRead_7/ReadVariableOpЂRead_8/DisableCopyOnReadЂRead_8/ReadVariableOpЂRead_9/DisableCopyOnReadЂRead_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: q
Read/DisableCopyOnReadDisableCopyOnReadread_disablecopyonread_variable"/device:CPU:0*
_output_shapes
 
Read/ReadVariableOpReadVariableOpread_disablecopyonread_variable^Read/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	a
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: Y

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0	*
_output_shapes
: 
Read_1/DisableCopyOnReadDisableCopyOnReadAread_1_disablecopyonread_qrnnnetwork_encodingnetwork_dense_kernel"/device:CPU:0*
_output_shapes
 Т
Read_1/ReadVariableOpReadVariableOpAread_1_disablecopyonread_qrnnnetwork_encodingnetwork_dense_kernel^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	б*
dtype0n

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	бd

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:	б
Read_2/DisableCopyOnReadDisableCopyOnRead?read_2_disablecopyonread_qrnnnetwork_encodingnetwork_dense_bias"/device:CPU:0*
_output_shapes
 М
Read_2/ReadVariableOpReadVariableOp?read_2_disablecopyonread_qrnnnetwork_encodingnetwork_dense_bias^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:б*
dtype0j

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:б`

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes	
:б
Read_3/DisableCopyOnReadDisableCopyOnReadCread_3_disablecopyonread_qrnnnetwork_encodingnetwork_dense_1_kernel"/device:CPU:0*
_output_shapes
 Х
Read_3/ReadVariableOpReadVariableOpCread_3_disablecopyonread_qrnnnetwork_encodingnetwork_dense_1_kernel^Read_3/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
б*
dtype0o

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
бe

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0* 
_output_shapes
:
б
Read_4/DisableCopyOnReadDisableCopyOnReadAread_4_disablecopyonread_qrnnnetwork_encodingnetwork_dense_1_bias"/device:CPU:0*
_output_shapes
 О
Read_4/ReadVariableOpReadVariableOpAread_4_disablecopyonread_qrnnnetwork_encodingnetwork_dense_1_bias^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:*
dtype0j

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:`

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes	
:
Read_5/DisableCopyOnReadDisableCopyOnRead:read_5_disablecopyonread_qrnnnetwork_dynamic_unroll_kernel"/device:CPU:0*
_output_shapes
 М
Read_5/ReadVariableOpReadVariableOp:read_5_disablecopyonread_qrnnnetwork_dynamic_unroll_kernel^Read_5/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
Д*
dtype0p
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
Дg
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0* 
_output_shapes
:
Д
Read_6/DisableCopyOnReadDisableCopyOnReadDread_6_disablecopyonread_qrnnnetwork_dynamic_unroll_recurrent_kernel"/device:CPU:0*
_output_shapes
 Х
Read_6/ReadVariableOpReadVariableOpDread_6_disablecopyonread_qrnnnetwork_dynamic_unroll_recurrent_kernel^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	MД*
dtype0o
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	MДf
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes
:	MД
Read_7/DisableCopyOnReadDisableCopyOnRead8read_7_disablecopyonread_qrnnnetwork_dynamic_unroll_bias"/device:CPU:0*
_output_shapes
 Е
Read_7/ReadVariableOpReadVariableOp8read_7_disablecopyonread_qrnnnetwork_dynamic_unroll_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:Д*
dtype0k
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Дb
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes	
:Д
Read_8/DisableCopyOnReadDisableCopyOnRead3read_8_disablecopyonread_qrnnnetwork_dense_2_kernel"/device:CPU:0*
_output_shapes
 Д
Read_8/ReadVariableOpReadVariableOp3read_8_disablecopyonread_qrnnnetwork_dense_2_kernel^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	Mи*
dtype0o
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	Mиf
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes
:	Mи
Read_9/DisableCopyOnReadDisableCopyOnRead1read_9_disablecopyonread_qrnnnetwork_dense_2_bias"/device:CPU:0*
_output_shapes
 Ў
Read_9/ReadVariableOpReadVariableOp1read_9_disablecopyonread_qrnnnetwork_dense_2_bias^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:и*
dtype0k
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:иb
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes	
:и
Read_10/DisableCopyOnReadDisableCopyOnRead4read_10_disablecopyonread_qrnnnetwork_dense_3_kernel"/device:CPU:0*
_output_shapes
 З
Read_10/ReadVariableOpReadVariableOp4read_10_disablecopyonread_qrnnnetwork_dense_3_kernel^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	иd*
dtype0p
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	иdf
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes
:	иd
Read_11/DisableCopyOnReadDisableCopyOnRead2read_11_disablecopyonread_qrnnnetwork_dense_3_bias"/device:CPU:0*
_output_shapes
 А
Read_11/ReadVariableOpReadVariableOp2read_11_disablecopyonread_qrnnnetwork_dense_3_bias^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:d*
dtype0k
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:da
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
:d
Read_12/DisableCopyOnReadDisableCopyOnReadEread_12_disablecopyonread_qrnnnetwork_num_action_project_dense_kernel"/device:CPU:0*
_output_shapes
 Ч
Read_12/ReadVariableOpReadVariableOpEread_12_disablecopyonread_qrnnnetwork_num_action_project_dense_kernel^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:d*
dtype0o
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:de
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes

:d
Read_13/DisableCopyOnReadDisableCopyOnReadCread_13_disablecopyonread_qrnnnetwork_num_action_project_dense_bias"/device:CPU:0*
_output_shapes
 С
Read_13/ReadVariableOpReadVariableOpCread_13_disablecopyonread_qrnnnetwork_num_action_project_dense_bias^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
:
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Г
valueЉBІB%train_step/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/0/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/1/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/2/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/3/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/4/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/5/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/6/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/7/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/8/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/9/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/10/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/11/.ATTRIBUTES/VARIABLE_VALUEB-model_variables/12/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B B B B B B B B B B B B B B B 
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtypes
2	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:Г
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_28Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_29IdentityIdentity_28:output:0^NoOp*
T0*
_output_shapes
: 
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "#
identity_29Identity_29:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:=9

_output_shapes
: 

_user_specified_nameConst:IE
C
_user_specified_name+)QRnnNetwork/num_action_project/dense/bias:KG
E
_user_specified_name-+QRnnNetwork/num_action_project/dense/kernel:84
2
_user_specified_nameQRnnNetwork/dense_3/bias::6
4
_user_specified_nameQRnnNetwork/dense_3/kernel:8
4
2
_user_specified_nameQRnnNetwork/dense_2/bias::	6
4
_user_specified_nameQRnnNetwork/dense_2/kernel:?;
9
_user_specified_name!QRnnNetwork/dynamic_unroll/bias:KG
E
_user_specified_name-+QRnnNetwork/dynamic_unroll/recurrent_kernel:A=
;
_user_specified_name#!QRnnNetwork/dynamic_unroll/kernel:HD
B
_user_specified_name*(QRnnNetwork/EncodingNetwork/dense_1/bias:JF
D
_user_specified_name,*QRnnNetwork/EncodingNetwork/dense_1/kernel:FB
@
_user_specified_name(&QRnnNetwork/EncodingNetwork/dense/bias:HD
B
_user_specified_name*(QRnnNetwork/EncodingNetwork/dense/kernel:($
"
_user_specified_name
Variable:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ц
а
<__inference_signature_wrapper_function_with_signature_294223
discount
observation

reward
	step_type
unknown
	unknown_0
	unknown_1:	б
	unknown_2:	б
	unknown_3:
б
	unknown_4:	
	unknown_5:
Д
	unknown_6:	MД
	unknown_7:	Д
	unknown_8:	Mи
	unknown_9:	и

unknown_10:	иd

unknown_11:d

unknown_12:d

unknown_13:
identity

identity_1

identity_2ЂStatefulPartitionedCallЖ
StatefulPartitionedCallStatefulPartitionedCall	step_typerewarddiscountobservationunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13*
Tin
2*
Tout
2*
_collective_manager_ids
 *I
_output_shapes7
5:џџџџџџџџџ:џџџџџџџџџM:џџџџџџџџџM*/
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *3
f.R,
*__inference_function_with_signature_294182k
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*#
_output_shapes
:џџџџџџџџџq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:џџџџџџџџџMq

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:џџџџџџџџџM<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџM:џџџџџџџџџM: : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name294215:&"
 
_user_specified_name294213:&"
 
_user_specified_name294211:&"
 
_user_specified_name294209:&"
 
_user_specified_name294207:&"
 
_user_specified_name294205:&"
 
_user_specified_name294203:&"
 
_user_specified_name294201:&
"
 
_user_specified_name294199:&	"
 
_user_specified_name294197:&"
 
_user_specified_name294195:&"
 
_user_specified_name294193:&"
 
_user_specified_name294191:LH
'
_output_shapes
:џџџџџџџџџM

_user_specified_name1/1:LH
'
_output_shapes
:џџџџџџџџџM

_user_specified_name1/0:PL
#
_output_shapes
:џџџџџџџџџ
%
_user_specified_name0/step_type:MI
#
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
0/reward:VR
'
_output_shapes
:џџџџџџџџџ
'
_user_specified_name0/observation:O K
#
_output_shapes
:џџџџџџџџџ
$
_user_specified_name
0/discount
и
c
"__inference_per_field_where_293938
t
f

sub_rank_1
shape_equal_1

identityY
assert_rank_at_least/ShapeShapet*
T0*
_output_shapes
::эЯa
Cassert_rank_at_least/assert_type/statically_determined_correct_typeNoOp*
_output_shapes
 R
4assert_rank_at_least/static_checks_determined_all_okNoOp*
_output_shapes
 F
RankConst*
_output_shapes
: *
dtype0*
value	B :F
subSubRank:output:0
sub_rank_1*
T0*
_output_shapes
: P
ShapeShapeshape_equal_1*
T0
*
_output_shapes
::эЯe
ones/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
џџџџџџџџџb
ones/ReshapeReshapesub:z:0ones/Reshape/shape:output:0*
T0*
_output_shapes
:L

ones/ConstConst*
_output_shapes
: *
dtype0*
value	B :f
onesFillones/Reshape:output:0ones/Const:output:0*
T0*#
_output_shapes
:џџџџџџџџџM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ~
concatConcatV2Shape:output:0ones:output:0concat/axis:output:0*
N*
T0*#
_output_shapes
:џџџџџџџџџU
ReshapeReshapeshape_equal_1concat:output:0*
T0
*
_output_shapes
:O
SelectV2SelectV2Reshape:output:0tf*
T0*
_output_shapes
:J
IdentityIdentitySelectV2:output:0*
T0*
_output_shapes
:"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:џџџџџџџџџM:џџџџџџџџџM: :џџџџџџџџџ:LH
#
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	Equal_1:>:

_output_shapes
: 
 
_user_specified_nameRank_1:JF
'
_output_shapes
:џџџџџџџџџM

_user_specified_namef:J F
'
_output_shapes
:џџџџџџџџџM

_user_specified_namet
е
j
*__inference_function_with_signature_294260
unknown:	 
identity	ЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallunknown*
Tin
2*
Tout
2	*
_collective_manager_ids
 *
_output_shapes
: *#
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *$
fR
__inference_<lambda>_293279^
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0	*
_output_shapes
: <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_name294256"эL
saver_filename:0StatefulPartitionedCall_2:0StatefulPartitionedCall_38"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*
action
4

0/discount&
action_0_discount:0џџџџџџџџџ
>
0/observation-
action_0_observation:0џџџџџџџџџ
0
0/reward$
action_0_reward:0џџџџџџџџџ
6
0/step_type'
action_0_step_type:0џџџџџџџџџ
*
1/0#
action_1_0:0џџџџџџџџџM
*
1/1#
action_1_1:0џџџџџџџџџM6
action,
StatefulPartitionedCall:0џџџџџџџџџ;
state/00
StatefulPartitionedCall:1џџџџџџџџџM;
state/10
StatefulPartitionedCall:2џџџџџџџџџMtensorflow/serving/predict*Ф
get_initial_stateЎ
2

batch_size$
get_initial_state_batch_size:0 -
0(
PartitionedCall:0џџџџџџџџџM-
1(
PartitionedCall:1џџџџџџџџџMtensorflow/serving/predict*,
get_metadatatensorflow/serving/predict*Z
get_train_stepH*
int64!
StatefulPartitionedCall_1:0	 tensorflow/serving/predict:Ѕ
ф
policy_state_spec

train_step
metadata
model_variables
_all_assets

action
distribution
get_initial_state
	get_metadata

get_train_step

signatures"
_generic_user_object
 "
trackable_list_wrapper
:	 (2Variable
 "
trackable_dict_wrapper

0
1
2
3
4
5
6
7
8
9
10
11
12"
trackable_tuple_wrapper
d
_policy_state_spec
_policy_step_spec
_wrapped_policy"
trackable_dict_wrapper
­
trace_0
trace_12і
__inference_action_294568
__inference_action_294864Н
ЖВВ
FullArgSpec0
args(%
j	time_step
jpolicy_state
jseed
varargs
 
varkw
 
defaultsЂ	
Ђ 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0ztrace_1
я
trace_02в
"__inference_distribution_fn_295133Ћ
ЄВ 
FullArgSpec(
args 
j	time_step
jpolicy_state
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
т
trace_02Х
$__inference_get_initial_state_295149
В
FullArgSpec
args
j
batch_size
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
ВBЏ
__inference_<lambda>_293281"
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
ВBЏ
__inference_<lambda>_293279"
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
`

 action
!get_initial_state
"get_train_step
#get_metadata"
signature_map
;:9	б2(QRnnNetwork/EncodingNetwork/dense/kernel
5:3б2&QRnnNetwork/EncodingNetwork/dense/bias
>:<
б2*QRnnNetwork/EncodingNetwork/dense_1/kernel
7:52(QRnnNetwork/EncodingNetwork/dense_1/bias
5:3
Д2!QRnnNetwork/dynamic_unroll/kernel
>:<	MД2+QRnnNetwork/dynamic_unroll/recurrent_kernel
.:,Д2QRnnNetwork/dynamic_unroll/bias
-:+	Mи2QRnnNetwork/dense_2/kernel
':%и2QRnnNetwork/dense_2/bias
-:+	иd2QRnnNetwork/dense_3/kernel
&:$d2QRnnNetwork/dense_3/bias
=:;d2+QRnnNetwork/num_action_project/dense/kernel
7:52)QRnnNetwork/num_action_project/dense/bias
 "
trackable_list_wrapper
3
	state
1"
trackable_tuple_wrapper
]
$
_q_network
%_policy_state_spec
&_policy_step_spec"
_generic_user_object
B
__inference_action_294568	step_typerewarddiscountobservation01"Г
ЌВЈ
FullArgSpec0
args(%
j	time_step
jpolicy_state
jseed
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ШBХ
__inference_action_294864time_step_step_typetime_step_rewardtime_step_discounttime_step_observationpolicy_state_0policy_state_1"Г
ЌВЈ
FullArgSpec0
args(%
j	time_step
jpolicy_state
jseed
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
"__inference_distribution_fn_295133	step_typerewarddiscountobservation01"Ћ
ЄВ 
FullArgSpec(
args 
j	time_step
jpolicy_state
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
жBг
$__inference_get_initial_state_295149
batch_size"
В
FullArgSpec
args
j
batch_size
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
юBы
<__inference_signature_wrapper_function_with_signature_294223
0/discount0/observation0/reward0/step_type1/01/1"ю
чВу
FullArgSpec
args 
varargs
 
varkw
 
defaults
 q

kwonlyargsc`
jarg_0_discount
jarg_0_observation
jarg_0_reward
jarg_0_step_type
	jarg_1_0
	jarg_1_1
kwonlydefaults
 
annotationsЊ *
 
юBы
<__inference_signature_wrapper_function_with_signature_294254
batch_size"
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs
j
batch_size
kwonlydefaults
 
annotationsЊ *
 
гBа
<__inference_signature_wrapper_function_with_signature_294267"
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
гBа
<__inference_signature_wrapper_function_with_signature_294272"
В
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ 
ђ
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses
-_state_spec
._input_encoder
/_lstm_network
0_output_encoder"
_tf_keras_layer
 "
trackable_list_wrapper
3
	%state
%1"
trackable_tuple_wrapper
~
0
1
2
3
4
5
6
7
8
9
10
11
12"
trackable_list_wrapper
~
0
1
2
3
4
5
6
7
8
9
10
11
12"
trackable_list_wrapper
 "
trackable_list_wrapper
­
1non_trainable_variables

2layers
3metrics
4layer_regularization_losses
5layer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses"
_generic_user_object
з2дб
ЪВЦ
FullArgSpecD
args<9
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaultsЂ	
Ђ 
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
з2дб
ЪВЦ
FullArgSpecD
args<9
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaultsЂ	
Ђ 
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
С
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses
<_postprocessing_layers"
_tf_keras_layer
Џ
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses
Ccell"
_tf_keras_layer
5
D0
E1
F2"
trackable_list_wrapper
 "
trackable_list_wrapper
C
.0
/1
D2
E3
F4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
<
0
1
2
3"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Gnon_trainable_variables

Hlayers
Imetrics
Jlayer_regularization_losses
Klayer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses"
_generic_user_object
л2ие
ЮВЪ
FullArgSpecD
args<9
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaultsЂ

 
Ђ 
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
л2ие
ЮВЪ
FullArgSpecD
args<9
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaultsЂ

 
Ђ 
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
5
L0
M1
N2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Onon_trainable_variables

Players
Qmetrics
Rlayer_regularization_losses
Slayer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses"
_generic_user_object
ж2га
ЩВХ
FullArgSpec@
args85
jinputs
jinitial_state
j
reset_mask

jtraining
varargs
 
varkw
 
defaultsЂ

 

 
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ж2га
ЩВХ
FullArgSpec@
args85
jinputs
jinitial_state
j
reset_mask

jtraining
varargs
 
varkw
 
defaultsЂ

 

 
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ј
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
X__call__
*Y&call_and_return_all_conditional_losses
Z_random_generator
[
state_size

kernel
recurrent_kernel
bias"
_tf_keras_layer
Л
\	variables
]trainable_variables
^regularization_losses
_	keras_api
`__call__
*a&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
Л
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
Л
h	variables
itrainable_variables
jregularization_losses
k	keras_api
l__call__
*m&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
 "
trackable_list_wrapper
5
L0
M1
N2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
Ѕ
n	variables
otrainable_variables
pregularization_losses
q	keras_api
r__call__
*s&call_and_return_all_conditional_losses"
_tf_keras_layer
Л
t	variables
utrainable_variables
vregularization_losses
w	keras_api
x__call__
*y&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
Л
z	variables
{trainable_variables
|regularization_losses
}	keras_api
~__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
 "
trackable_list_wrapper
'
C0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
5
0
1
2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
T	variables
Utrainable_variables
Vregularization_losses
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses"
_generic_user_object
Й2ЖГ
ЌВЈ
FullArgSpec+
args# 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Й2ЖГ
ЌВЈ
FullArgSpec+
args# 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
"
_generic_user_object
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
\	variables
]trainable_variables
^regularization_losses
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses"
_generic_user_object
2
В
FullArgSpec
args

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
annotationsЊ *
 
2
В
FullArgSpec
args

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
annotationsЊ *
 
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses"
_generic_user_object
2
В
FullArgSpec
args

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
annotationsЊ *
 
2
В
FullArgSpec
args

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
annotationsЊ *
 
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
h	variables
itrainable_variables
jregularization_losses
l__call__
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses"
_generic_user_object
2
В
FullArgSpec
args

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
annotationsЊ *
 
2
В
FullArgSpec
args

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
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
n	variables
otrainable_variables
pregularization_losses
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses"
_generic_user_object
2
В
FullArgSpec
args

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
annotationsЊ *
 
2
В
FullArgSpec
args

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
annotationsЊ *
 
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
t	variables
utrainable_variables
vregularization_losses
x__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object
2
В
FullArgSpec
args

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
annotationsЊ *
 
2
В
FullArgSpec
args

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
annotationsЊ *
 
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
В
non_trainable_variables
layers
 metrics
 Ёlayer_regularization_losses
Ђlayer_metrics
z	variables
{trainable_variables
|regularization_losses
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
2
В
FullArgSpec
args

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
annotationsЊ *
 
2
В
FullArgSpec
args

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
annotationsЊ *
 
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
trackable_dict_wrapperC
__inference_<lambda>_293279$Ђ

Ђ 
Њ "
unknown 	3
__inference_<lambda>_293281Ђ

Ђ 
Њ "Њ ш
__inference_action_294568ЪЂ
Ђ
ЦВТ
TimeStep,
	step_type
	step_typeџџџџџџџџџ&
reward
rewardџџџџџџџџџ*
discount
discountџџџџџџџџџ4
observation%"
observationџџџџџџџџџ
=:

0џџџџџџџџџM

1џџџџџџџџџM

 
Њ "В

PolicyStep&
action
actionџџџџџџџџџR
stateIF
!
state_0џџџџџџџџџM
!
state_1џџџџџџџџџM
infoЂ Њ
__inference_action_294864оЂк
вЂЮ
юВъ
TimeStep6
	step_type)&
time_step_step_typeџџџџџџџџџ0
reward&#
time_step_rewardџџџџџџџџџ4
discount(%
time_step_discountџџџџџџџџџ>
observation/,
time_step_observationџџџџџџџџџ
WT
(%
policy_state_0џџџџџџџџџM
(%
policy_state_1џџџџџџџџџM

 
Њ "В

PolicyStep&
action
actionџџџџџџџџџR
stateIF
!
state_0џџџџџџџџџM
!
state_1џџџџџџџџџM
infoЂ Ъ
"__inference_distribution_fn_295133ЃЂ
Ђ
ЦВТ
TimeStep,
	step_type
	step_typeџџџџџџџџџ&
reward
rewardџџџџџџџџџ*
discount
discountџџџџџџџџџ4
observation%"
observationџџџџџџџџџ
=:

0џџџџџџџџџM

1џџџџџџџџџM
Њ "іВђ

PolicyStep
actionїѓПЂЛ
`
BЊ?

atol 

locџџџџџџџџџ

rtol 
LЊI

allow_nan_statsp

namejDeterministic_1_1

validate_argsp 
Ђ
j
parameters
Ђ 
Ђ
jname+tfp.distributions.Deterministic_ACTTypeSpec R
stateIF
!
state_0џџџџџџџџџM
!
state_1џџџџџџџџџM
infoЂ 
$__inference_get_initial_state_295149q"Ђ
Ђ


batch_size 
Њ "KH
"
tensor_0џџџџџџџџџM
"
tensor_1џџџџџџџџџM
<__inference_signature_wrapper_function_with_signature_294223йМЂИ
Ђ 
АЊЌ
2
arg_0_discount 

0/discountџџџџџџџџџ
<
arg_0_observation'$
0/observationџџџџџџџџџ
.
arg_0_reward
0/rewardџџџџџџџџџ
4
arg_0_step_type!
0/step_typeџџџџџџџџџ
(
arg_1_0
1/0џџџџџџџџџM
(
arg_1_1
1/1џџџџџџџџџM"Њ
&
action
actionџџџџџџџџџ
,
state/0!
state_0џџџџџџџџџM
,
state/1!
state_1џџџџџџџџџMЪ
<__inference_signature_wrapper_function_with_signature_2942540Ђ-
Ђ 
&Њ#
!

batch_size

batch_size "UЊR
'
0"
tensor_0џџџџџџџџџM
'
1"
tensor_1џџџџџџџџџMp
<__inference_signature_wrapper_function_with_signature_2942670Ђ

Ђ 
Њ "Њ

int64
int64 	T
<__inference_signature_wrapper_function_with_signature_294272Ђ

Ђ 
Њ "Њ 