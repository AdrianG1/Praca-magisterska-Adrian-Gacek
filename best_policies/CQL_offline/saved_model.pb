ау
ы!╗!
D
AddV2
x"T
y"T
z"T"
Ttype:
2	АР
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( И
А
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
resourceИ
R
Equal
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(Р
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
о
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
Ж
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( И
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
2	Р

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
│
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
П
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( ""
Ttype:
2	"
Tidxtype0:
2	
Е
RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	И
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
dtypetypeИ
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
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
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
d
Shape

input"T&
output"out_typeКэout_type"	
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
@
Softplus
features"T
activations"T"
Ttype:
2
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
┴
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
executor_typestring Ии
@
StaticRegexFullMatch	
input

output
"
patternstring
ў
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
░
VarHandleOp
resource"
	containerstring "
shared_namestring "

debug_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И
&
	ZerosLike
x"T
y"T"	
Ttype"serve*2.15.02v2.15.0-rc1-8-g6887368d6d48╫С
╪
OActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/biasVarHandleOp*
_output_shapes
: *`

debug_nameRPActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/bias/*
dtype0*
shape:*`
shared_nameQOActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/bias
я
cActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/bias/Read/ReadVariableOpReadVariableOpOActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/bias*
_output_shapes
:*
dtype0
т
QActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/kernelVarHandleOp*
_output_shapes
: *b

debug_nameTRActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/kernel/*
dtype0*
shape
:d*b
shared_nameSQActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/kernel
ў
eActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/kernel/Read/ReadVariableOpReadVariableOpQActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/kernel*
_output_shapes

:d*
dtype0
┤
CActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/biasVarHandleOp*
_output_shapes
: *T

debug_nameFDActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/bias/*
dtype0*
shape:*T
shared_nameECActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/bias
╫
WActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/bias/Read/ReadVariableOpReadVariableOpCActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/bias*
_output_shapes
:*
dtype0
╖
DActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/biasVarHandleOp*
_output_shapes
: *U

debug_nameGEActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/bias/*
dtype0*
shape:d*U
shared_nameFDActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/bias
┘
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/bias/Read/ReadVariableOpReadVariableOpDActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/bias*
_output_shapes
:d*
dtype0
┴
FActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/kernelVarHandleOp*
_output_shapes
: *W

debug_nameIGActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/kernel/*
dtype0*
shape
:gd*W
shared_nameHFActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/kernel
с
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/kernel/Read/ReadVariableOpReadVariableOpFActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/kernel*
_output_shapes

:gd*
dtype0
═
KActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/biasVarHandleOp*
_output_shapes
: *\

debug_nameNLActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/bias/*
dtype0*
shape:Ь*\
shared_nameMKActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/bias
ш
_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/bias/Read/ReadVariableOpReadVariableOpKActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/bias*
_output_shapes	
:Ь*
dtype0
ї
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/recurrent_kernelVarHandleOp*
_output_shapes
: *h

debug_nameZXActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/recurrent_kernel/*
dtype0*
shape:	gЬ*h
shared_nameYWActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/recurrent_kernel
Д
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/recurrent_kernel/Read/ReadVariableOpReadVariableOpWActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/recurrent_kernel*
_output_shapes
:	gЬ*
dtype0
╪
MActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/kernelVarHandleOp*
_output_shapes
: *^

debug_namePNActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/kernel/*
dtype0*
shape:
ЖЬ*^
shared_nameOMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/kernel
ё
aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/kernel/Read/ReadVariableOpReadVariableOpMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/kernel* 
_output_shapes
:
ЖЬ*
dtype0
т
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/biasVarHandleOp*
_output_shapes
: *c

debug_nameUSActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/bias/*
dtype0*
shape:Ж*c
shared_nameTRActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/bias
Ў
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/bias/Read/ReadVariableOpReadVariableOpRActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/bias*
_output_shapes	
:Ж*
dtype0
ь
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/kernelVarHandleOp*
_output_shapes
: *e

debug_nameWUActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/kernel/*
dtype0*
shape:	Ж*e
shared_nameVTActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/kernel
■
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/kernel/Read/ReadVariableOpReadVariableOpTActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/kernel*
_output_shapes
:	Ж*
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
:         *
dtype0*
shape:         
w
action_0_observationPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
j
action_0_rewardPlaceholder*#
_output_shapes
:         *
dtype0*
shape:         
m
action_0_step_typePlaceholder*#
_output_shapes
:         *
dtype0*
shape:         
m

action_1_0Placeholder*'
_output_shapes
:         g*
dtype0*
shape:         g
m

action_1_1Placeholder*'
_output_shapes
:         g*
dtype0*
shape:         g
э	
StatefulPartitionedCallStatefulPartitionedCallaction_0_discountaction_0_observationaction_0_rewardaction_0_step_type
action_1_0
action_1_1TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/kernelRActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/biasMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/kernelWActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/recurrent_kernelKActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/biasFActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/kernelDActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/biasQActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/kernelOActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/biasCActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *I
_output_shapes7
5:         :         g:         g*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8В *E
f@R>
<__inference_signature_wrapper_function_with_signature_435797
]
get_initial_state_batch_sizePlaceholder*
_output_shapes
: *
dtype0*
shape: 
╘
PartitionedCallPartitionedCallget_initial_state_batch_size*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:         g:         g* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *E
f@R>
<__inference_signature_wrapper_function_with_signature_435828
ї
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
GPU2*0J 8В *E
f@R>
<__inference_signature_wrapper_function_with_signature_435846
░
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
GPU2*0J 8В *E
f@R>
<__inference_signature_wrapper_function_with_signature_435841

NoOpNoOp
∙A
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*┤A
valueкABзA BаA
╩
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
J
0
1
2
3
4
5
6
7
8
9*
G
_actor_network
_policy_state_spec
_policy_step_spec*

trace_0
trace_1* 

trace_0* 

trace_0* 
* 
* 
K

action
get_initial_state
get_train_step
 get_metadata* 
ЫФ
VARIABLE_VALUETActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/kernel,model_variables/0/.ATTRIBUTES/VARIABLE_VALUE*
ЩТ
VARIABLE_VALUERActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/bias,model_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
ФН
VARIABLE_VALUEMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/kernel,model_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
ЮЧ
VARIABLE_VALUEWActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/recurrent_kernel,model_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
ТЛ
VARIABLE_VALUEKActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/bias,model_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
НЖ
VARIABLE_VALUEFActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/kernel,model_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
ЛД
VARIABLE_VALUEDActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/bias,model_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
КГ
VARIABLE_VALUECActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/bias,model_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
ШС
VARIABLE_VALUEQActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/kernel,model_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
ЦП
VARIABLE_VALUEOActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/bias,model_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
╬
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses
'_state_spec
(_lstm_encoder
)_projection_networks*
* 

	state
1* 
* 
* 
* 
* 
* 
* 
* 
* 
J
0
1
2
3
4
5
6
7
8
9*
J
0
1
2
3
4
5
6
7
8
9*
* 
У
*non_trainable_variables

+layers
,metrics
-layer_regularization_losses
.layer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses*
* 
* 
* 
▌
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses
5_state_spec
6_input_encoder
7_lstm_network
8_output_encoder*
╕
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses
?_means_projection_layer
	@_bias*
* 

(0
)1*
* 
* 
* 
5
0
1
2
3
4
5
6*
5
0
1
2
3
4
5
6*
* 
У
Anon_trainable_variables

Blayers
Cmetrics
Dlayer_regularization_losses
Elayer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses*
* 
* 
* 
м
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
J__call__
*K&call_and_return_all_conditional_losses
L_postprocessing_layers*
Ъ
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q__call__
*R&call_and_return_all_conditional_losses
Scell*

T0*

0
1
2*

0
1
2*
* 
У
Unon_trainable_variables

Vlayers
Wmetrics
Xlayer_regularization_losses
Ylayer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses*
* 
* 
ж
Z	variables
[trainable_variables
\regularization_losses
]	keras_api
^__call__
*_&call_and_return_all_conditional_losses

kernel
bias*
Ъ
`	variables
atrainable_variables
bregularization_losses
c	keras_api
d__call__
*e&call_and_return_all_conditional_losses
bias*
* 

60
71
T2*
* 
* 
* 

0
1*

0
1*
* 
У
fnon_trainable_variables

glayers
hmetrics
ilayer_regularization_losses
jlayer_metrics
F	variables
Gtrainable_variables
Hregularization_losses
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses*
* 
* 

k0
l1*

0
1
2*

0
1
2*
* 
У
mnon_trainable_variables

nlayers
ometrics
player_regularization_losses
qlayer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses*
* 
* 
у
r	variables
strainable_variables
tregularization_losses
u	keras_api
v__call__
*w&call_and_return_all_conditional_losses
x_random_generator
y
state_size

kernel
recurrent_kernel
bias*
ж
z	variables
{trainable_variables
|regularization_losses
}	keras_api
~__call__
*&call_and_return_all_conditional_losses

kernel
bias*
* 

?0
@1*
* 
* 
* 

0
1*

0
1*
* 
Ш
Аnon_trainable_variables
Бlayers
Вmetrics
 Гlayer_regularization_losses
Дlayer_metrics
Z	variables
[trainable_variables
\regularization_losses
^__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses*
* 
* 

0*

0*
* 
Ш
Еnon_trainable_variables
Жlayers
Зmetrics
 Иlayer_regularization_losses
Йlayer_metrics
`	variables
atrainable_variables
bregularization_losses
d__call__
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses*
* 
* 
* 

k0
l1*
* 
* 
* 
Ф
К	variables
Лtrainable_variables
Мregularization_losses
Н	keras_api
О__call__
+П&call_and_return_all_conditional_losses* 
м
Р	variables
Сtrainable_variables
Тregularization_losses
У	keras_api
Ф__call__
+Х&call_and_return_all_conditional_losses

kernel
bias*
* 

S0*
* 
* 
* 

0
1
2*

0
1
2*
* 
Ш
Цnon_trainable_variables
Чlayers
Шmetrics
 Щlayer_regularization_losses
Ъlayer_metrics
r	variables
strainable_variables
tregularization_losses
v__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses*
* 
* 
* 
* 

0
1*

0
1*
* 
Ш
Ыnon_trainable_variables
Ьlayers
Эmetrics
 Юlayer_regularization_losses
Яlayer_metrics
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
Ь
аnon_trainable_variables
бlayers
вmetrics
 гlayer_regularization_losses
дlayer_metrics
К	variables
Лtrainable_variables
Мregularization_losses
О__call__
+П&call_and_return_all_conditional_losses
'П"call_and_return_conditional_losses* 
* 
* 

0
1*

0
1*
* 
Ю
еnon_trainable_variables
жlayers
зmetrics
 иlayer_regularization_losses
йlayer_metrics
Р	variables
Сtrainable_variables
Тregularization_losses
Ф__call__
+Х&call_and_return_all_conditional_losses
'Х"call_and_return_conditional_losses*
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
╔
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameVariableTActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/kernelRActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/biasMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/kernelWActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/recurrent_kernelKActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/biasFActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/kernelDActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/biasCActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/biasQActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/kernelOActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/biasConst*
Tin
2*
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
GPU2*0J 8В *(
f#R!
__inference__traced_save_436608
─
StatefulPartitionedCall_3StatefulPartitionedCallsaver_filenameVariableTActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/kernelRActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/biasMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/kernelWActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/recurrent_kernelKActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/biasFActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/kernelDActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/biasCActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/biasQActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/kernelOActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/bias*
Tin
2*
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
GPU2*0J 8В *+
f&R$
"__inference__traced_restore_436650╗ў
╜Ь
ш
__inference_action_435735
	time_step
time_step_1
time_step_2
time_step_3
policy_state
policy_state_1
lactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_matmul_readvariableop_resource:	Ж|
mactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_biasadd_readvariableop_resource:	ЖГ
oactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_matmul_readvariableop_resource:
ЖЬД
qactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_matmul_1_readvariableop_resource:	gЬ
pactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_biasadd_readvariableop_resource:	Ьs
aactordistributionrnnnetwork_actordistributionrnnnetwork_dense_1_tensordot_readvariableop_resource:gdm
_actordistributionrnnnetwork_actordistributionrnnnetwork_dense_1_biasadd_readvariableop_resource:d{
iactordistributionrnnnetwork_normalprojectionnetwork_means_projection_layer_matmul_readvariableop_resource:dx
jactordistributionrnnnetwork_normalprojectionnetwork_means_projection_layer_biasadd_readvariableop_resource:l
^actordistributionrnnnetwork_normalprojectionnetwork_bias_layer_biasadd_readvariableop_resource:
identity

identity_1

identity_2ИвdActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpвcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpвVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/BiasAdd/ReadVariableOpвXActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/ReadVariableOpвgActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOpвfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOpвhActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOpвUActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOpвaActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOpв`ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOpN
ShapeShapetime_step_2*
T0*
_output_shapes
::э╧]
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
valueB:╤
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
valueB:gM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Б
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
:         g[
shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB:gO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : З
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
:         gI
Equal/yConst*
_output_shapes
: *
dtype0*
value	B : Y
EqualEqual	time_stepEqual/y:output:0*
T0*#
_output_shapes
:         F
RankConst*
_output_shapes
: *
dtype0*
value	B :├
PartitionedCallPartitionedCallzeros:output:0policy_stateRank:output:0	Equal:z:0*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         g* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *+
f&R$
"__inference_per_field_where_435551╔
PartitionedCall_1PartitionedCallzeros_1:output:0policy_state_1Rank:output:0	Equal:z:0*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         g* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *+
f&R$
"__inference_per_field_where_435551И
FActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :ф
BActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims
ExpandDimstime_step_3OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims/dim:output:0*
T0*+
_output_shapes
:         К
HActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :т
DActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1
ExpandDims	time_stepQActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:         Ї
[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/ShapeShapeKActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	:э╨┤
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"       ╒
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/ReshapeReshapeKActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims:output:0lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape/shape:output:0*
T0*'
_output_shapes
:         ж
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ▄
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten/ReshapeReshapefActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape:output:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten/Const:output:0*
T0*'
_output_shapes
:         С
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpReadVariableOplactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_matmul_readvariableop_resource*
_output_shapes
:	Ж*
dtype0р
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMulMatMul`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten/Reshape:output:0kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ЖП
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpReadVariableOpmactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_biasadd_readvariableop_resource*
_output_shapes	
:Ж*
dtype0с
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAddBiasAdd^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul:product:0lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Жэ
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/ReluRelu^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd:output:0*
T0*(
_output_shapes
:         Ж╡
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ╖
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:╖
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Е
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_sliceStridedSlicedActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Shape:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack:output:0vActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_1:output:0vActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_maskЛ
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/ShapeShape`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/Relu:activations:0*
T0*
_output_shapes
:*
out_type0	:э╨╖
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:╣
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ╣
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Н
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1StridedSlicefActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/Shape:output:0vActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack:output:0xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_1:output:0xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
end_maskе
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ш
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concatConcatV2nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice:output:0pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1:output:0lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat/axis:output:0*
N*
T0	*
_output_shapes
:·
_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/ReshapeReshape`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/Relu:activations:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat:output:0*
Tshape0	*
T0*,
_output_shapes
:         ЖА
>ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask/yConst*
_output_shapes
: *
dtype0*
value	B : П
<ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/maskEqualMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1:output:0GActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask/y:output:0*
T0*'
_output_shapes
:         Н
KActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/RankConst*
_output_shapes
: *
dtype0*
value	B :Ф
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/startConst*
_output_shapes
: *
dtype0*
value	B :Ф
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :Б
LActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/rangeRange[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/start:output:0TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Rank:output:0[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/delta:output:0*
_output_shapes
:з
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB"       Ф
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ь
MActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concatConcatV2_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/values_0:output:0UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range:output:0[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/axis:output:0*
N*
T0*
_output_shapes
:╓
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose	TransposehActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/Reshape:output:0VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat:output:0*
T0*,
_output_shapes
:         Ж▐
LActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ShapeShapeTActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose:y:0*
T0*
_output_shapes
::э╧д
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:ж
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:ж
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┤
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_sliceStridedSliceUActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Shape:output:0cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack:output:0eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_1:output:0eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskи
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       ╡
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1	Transpose@ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask:z:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1/perm:output:0*
T0
*'
_output_shapes
:         Ч
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :g╚
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packedPack]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice:output:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:Ч
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ┴
LActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zerosFill\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packed:output:0[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/Const:output:0*
T0*'
_output_shapes
:         gЩ
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :g╠
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packedPack]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice:output:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Щ
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ╟
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1Fill^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packed:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/Const:output:0*
T0*'
_output_shapes
:         g∙
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/SqueezeSqueezeTActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose:y:0*
T0*(
_output_shapes
:         Ж*
squeeze_dims
 °
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze_1SqueezeVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1:y:0*
T0
*#
_output_shapes
:         *
squeeze_dims
 ╒
MActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/SelectSelectYActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze_1:output:0UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros:output:0PartitionedCall:output:0*
T0*'
_output_shapes
:         g█
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Select_1SelectYActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze_1:output:0WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1:output:0PartitionedCall_1:output:0*
T0*'
_output_shapes
:         gШ
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOpReadVariableOpoactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_matmul_readvariableop_resource* 
_output_shapes
:
ЖЬ*
dtype0▌
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMulMatMulWActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze:output:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ЬЫ
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpqactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes
:	gЬ*
dtype0р
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1MatMulVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Select:output:0pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ь╪
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/addAddV2aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul:product:0cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:         ЬХ
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOpReadVariableOppactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:Ь*
dtype0с
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAddBiasAddXActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/add:z:0oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ьв
`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :й
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/splitSplitiActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split/split_dim:output:0aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:         g:         g:         g:         g*
	num_splitЎ
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/SigmoidSigmoid_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split:output:0*
T0*'
_output_shapes
:         g°
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_1Sigmoid_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split:output:1*
T0*'
_output_shapes
:         g╟
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mulMul^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_1:y:0XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Select_1:output:0*
T0*'
_output_shapes
:         gЁ
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/TanhTanh_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split:output:2*
T0*'
_output_shapes
:         g╚
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul_1Mul\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid:y:0YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:         g╟
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/add_1AddV2XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul:z:0ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:         g°
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_2Sigmoid_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split:output:3*
T0*'
_output_shapes
:         gэ
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Tanh_1TanhZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:         g╠
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul_2Mul^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_2:y:0[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:         gЧ
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :╤
QActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims
ExpandDimsZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul_2:z:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims/dim:output:0*
T0*+
_output_shapes
:         g·
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/ReadVariableOpReadVariableOpaactordistributionrnnnetwork_actordistributionrnnnetwork_dense_1_tensordot_readvariableop_resource*
_output_shapes

:gd*
dtype0Ш
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:Я
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ч
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/ShapeShapeZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims:output:0*
T0*
_output_shapes
::э╧Щ
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╗
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/GatherV2GatherV2XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/Shape:output:0WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/free:output:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Ы
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ┐
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/GatherV2_1GatherV2XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/Shape:output:0WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/axes:output:0bActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Щ
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: о
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/ProdProd[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/GatherV2:output:0XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: Ы
QActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ┤
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/Prod_1Prod]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/GatherV2_1:output:0ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: Ч
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ь
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/concatConcatV2WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/free:output:0WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/axes:output:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:╣
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/stackPackWActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/Prod:output:0YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:═
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/transpose	TransposeZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims:output:0YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:         g╩
QActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/ReshapeReshapeWActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/transpose:y:0XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  ╩
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/MatMulMatMulZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/Reshape:output:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         dЫ
QActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:dЩ
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : з
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/concat_1ConcatV2[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/GatherV2:output:0ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/Const_2:output:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:├
IActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/TensordotReshapeZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/MatMul:product:0[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         dЄ
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/BiasAdd/ReadVariableOpReadVariableOp_actordistributionrnnnetwork_actordistributionrnnnetwork_dense_1_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0╝
GActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/BiasAddBiasAddRActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot:output:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         d╘
DActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/ReluReluPActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/BiasAdd:output:0*
T0*+
_output_shapes
:         dч
?ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/SqueezeSqueezeRActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Relu:activations:0*
T0*'
_output_shapes
:         d*
squeeze_dims
К
`ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOpReadVariableOpiactordistributionrnnnetwork_normalprojectionnetwork_means_projection_layer_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0┴
QActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMulMatMulHActorDistributionRnnNetwork/ActorDistributionRnnNetwork/Squeeze:output:0hActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         И
aActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOpReadVariableOpjactordistributionrnnnetwork_normalprojectionnetwork_means_projection_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0╫
RActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAddBiasAdd[ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul:product:0iActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Ф
AActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         Э
;ActorDistributionRnnNetwork/NormalProjectionNetwork/ReshapeReshape[ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd:output:0JActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape/shape:output:0*
T0*#
_output_shapes
:         ┤
8ActorDistributionRnnNetwork/NormalProjectionNetwork/TanhTanhDActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape:output:0*
T0*#
_output_shapes
:         ~
9ActorDistributionRnnNetwork/NormalProjectionNetwork/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?ю
7ActorDistributionRnnNetwork/NormalProjectionNetwork/mulMulBActorDistributionRnnNetwork/NormalProjectionNetwork/mul/x:output:0<ActorDistributionRnnNetwork/NormalProjectionNetwork/Tanh:y:0*
T0*#
_output_shapes
:         ~
9ActorDistributionRnnNetwork/NormalProjectionNetwork/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?я
7ActorDistributionRnnNetwork/NormalProjectionNetwork/addAddV2BActorDistributionRnnNetwork/NormalProjectionNetwork/add/x:output:0;ActorDistributionRnnNetwork/NormalProjectionNetwork/mul:z:0*
T0*#
_output_shapes
:         ╢
>ActorDistributionRnnNetwork/NormalProjectionNetwork/zeros_like	ZerosLike;ActorDistributionRnnNetwork/NormalProjectionNetwork/add:z:0*
T0*#
_output_shapes
:         Ш
MActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
         е
IActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/ExpandDims
ExpandDimsBActorDistributionRnnNetwork/NormalProjectionNetwork/zeros_like:y:0VActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/ExpandDims/dim:output:0*
T0*'
_output_shapes
:         Ё
UActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOpReadVariableOp^actordistributionrnnnetwork_normalprojectionnetwork_bias_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0╢
FActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAddBiasAddRActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/ExpandDims:output:0]ActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         г
RActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        е
TActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       е
TActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ░
LActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/strided_sliceStridedSliceOActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd:output:0[ActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/strided_slice/stack:output:0]ActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/strided_slice/stack_1:output:0]ActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/strided_slice/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *
ellipsis_mask*
shrink_axis_maskЦ
CActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         Ы
=ActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape_1ReshapeUActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/strided_slice:output:0LActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape_1/shape:output:0*
T0*#
_output_shapes
:         ╛
<ActorDistributionRnnNetwork/NormalProjectionNetwork/SoftplusSoftplusFActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape_1:output:0*
T0*#
_output_shapes
:         а
]Normal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB ╧
VNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/ShapeShape;ActorDistributionRnnNetwork/NormalProjectionNetwork/add:z:0*
T0*
_output_shapes
::э╧Ш
VNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : о
dNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ░
fNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:░
fNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ф
^Normal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/strided_sliceStridedSlice_Normal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/Shape:output:0mNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/strided_slice/stack:output:0oNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/strided_slice/stack_1:output:0oNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskр
XNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/Shape_1ShapeJActorDistributionRnnNetwork/NormalProjectionNetwork/Softplus:activations:0*
T0*
_output_shapes
::э╧Ъ
XNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/Const_1Const*
_output_shapes
: *
dtype0*
value	B : ░
fNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ▓
hNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:▓
hNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ю
`Normal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/strided_slice_1StridedSliceaNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/Shape_1:output:0oNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/strided_slice_1/stack:output:0qNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/strided_slice_1/stack_1:output:0qNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskд
aNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB ж
cNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB т
^Normal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/BroadcastArgsBroadcastArgslNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/BroadcastArgs/s0_1:output:0gNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/strided_slice:output:0*
_output_shapes
:▌
`Normal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/BroadcastArgs_1BroadcastArgscNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/BroadcastArgs:r0:0iNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/strided_slice_1:output:0*
_output_shapes
:к
`Normal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:Ю
\Normal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ╩
WNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/concatConcatV2iNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/concat/values_0:output:0eNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/BroadcastArgs_1:r0:0eNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:п
jNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    ▒
lNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  А?│
zNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/normal/random_normal/RandomStandardNormalRandomStandardNormal`Normal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/concat:output:0*
T0*'
_output_shapes
:         *
dtype0Я
iNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/normal/random_normal/mulMulГNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/normal/random_normal/RandomStandardNormal:output:0uNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/normal/random_normal/stddev:output:0*
T0*'
_output_shapes
:         Д
eNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/normal/random_normalAddV2mNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/normal/random_normal/mul:z:0sNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/normal/random_normal/mean:output:0*
T0*'
_output_shapes
:         ─
TNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/mulMuliNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/normal/random_normal:z:0JActorDistributionRnnNetwork/NormalProjectionNetwork/Softplus:activations:0*
T0*'
_output_shapes
:         ж
TNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/addAddV2XNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/mul:z:0;ActorDistributionRnnNetwork/NormalProjectionNetwork/add:z:0*
T0*'
_output_shapes
:         б
^Normal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/Shape_2/ConstConst*
_output_shapes
: *
dtype0*
valueB в
XNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/Shape_2Const*
_output_shapes
:*
dtype0*
valueB: ю
XNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/Shape_3ShapeXNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/add:z:0*
T0*
_output_shapes
::э╧░
fNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:▓
hNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ▓
hNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ь
`Normal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/strided_slice_2StridedSliceaNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/Shape_3:output:0oNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/strided_slice_2/stack:output:0qNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/strided_slice_2/stack_1:output:0qNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskа
^Normal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ╧
YNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/concat_1ConcatV2fNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/sample_shape:output:0iNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/strided_slice_2:output:0gNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:╧
XNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/ReshapeReshapeXNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/add:z:0bNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/concat_1:output:0*
T0*#
_output_shapes
:         \
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╙
clip_by_value/MinimumMinimumaNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0*#
_output_shapes
:         T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    {
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*#
_output_shapes
:         \
IdentityIdentityclip_by_value:z:0^NoOp*
T0*#
_output_shapes
:         л

Identity_1IdentityZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul_2:z:0^NoOp*
T0*'
_output_shapes
:         gл

Identity_2IdentityZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/add_1:z:0^NoOp*
T0*'
_output_shapes
:         gА
NoOpNoOpe^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpd^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpW^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/BiasAdd/ReadVariableOpY^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/ReadVariableOph^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOpi^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOpV^ActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOpb^ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOpa^ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Н
_input_shapes|
z:         :         :         :         :         g:         g: : : : : : : : : : 2╠
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpdActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp2╩
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp2░
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/BiasAdd/ReadVariableOpVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/BiasAdd/ReadVariableOp2┤
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/ReadVariableOpXActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/ReadVariableOp2╥
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOpgActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOp2╨
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOpfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOp2╘
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOphActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOp2о
UActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOpUActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOp2╞
aActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOpaActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOp2─
`ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp`ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp:($
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
:         g
&
_user_specified_namepolicy_state:UQ
'
_output_shapes
:         g
&
_user_specified_namepolicy_state:RN
'
_output_shapes
:         
#
_user_specified_name	time_step:NJ
#
_output_shapes
:         
#
_user_specified_name	time_step:NJ
#
_output_shapes
:         
#
_user_specified_name	time_step:N J
#
_output_shapes
:         
#
_user_specified_name	time_step
р
Ї
<__inference_signature_wrapper_function_with_signature_435797
discount
observation

reward
	step_type
unknown
	unknown_0
	unknown_1:	Ж
	unknown_2:	Ж
	unknown_3:
ЖЬ
	unknown_4:	gЬ
	unknown_5:	Ь
	unknown_6:gd
	unknown_7:d
	unknown_8:d
	unknown_9:

unknown_10:
identity

identity_1

identity_2ИвStatefulPartitionedCallМ
StatefulPartitionedCallStatefulPartitionedCall	step_typerewarddiscountobservationunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *I
_output_shapes7
5:         :         g:         g*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8В *3
f.R,
*__inference_function_with_signature_435762k
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*#
_output_shapes
:         q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         gq

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:         g<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Н
_input_shapes|
z:         :         :         :         :         g:         g: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name435789:&"
 
_user_specified_name435787:&"
 
_user_specified_name435785:&"
 
_user_specified_name435783:&"
 
_user_specified_name435781:&
"
 
_user_specified_name435779:&	"
 
_user_specified_name435777:&"
 
_user_specified_name435775:&"
 
_user_specified_name435773:&"
 
_user_specified_name435771:LH
'
_output_shapes
:         g

_user_specified_name1/1:LH
'
_output_shapes
:         g

_user_specified_name1/0:PL
#
_output_shapes
:         
%
_user_specified_name0/step_type:MI
#
_output_shapes
:         
"
_user_specified_name
0/reward:VR
'
_output_shapes
:         
'
_user_specified_name0/observation:O K
#
_output_shapes
:         
$
_user_specified_name
0/discount
¤
T
$__inference_get_initial_state_435815

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
valueB:gM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Б
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
:         g[
shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB:gO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : З
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
:         gV
IdentityIdentityzeros:output:0*
T0*'
_output_shapes
:         gZ

Identity_1Identityzeros_1:output:0*
T0*'
_output_shapes
:         g"!

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
╜
т
*__inference_function_with_signature_435762
	step_type

reward
discount
observation
unknown
	unknown_0
	unknown_1:	Ж
	unknown_2:	Ж
	unknown_3:
ЖЬ
	unknown_4:	gЬ
	unknown_5:	Ь
	unknown_6:gd
	unknown_7:d
	unknown_8:d
	unknown_9:

unknown_10:
identity

identity_1

identity_2ИвStatefulPartitionedCall√
StatefulPartitionedCallStatefulPartitionedCall	step_typerewarddiscountobservationunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10*
Tin
2*
Tout
2*
_collective_manager_ids
 *I
_output_shapes7
5:         :         g:         g*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8В *"
fR
__inference_action_435735k
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*#
_output_shapes
:         q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         gq

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:         g<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Н
_input_shapes|
z:         :         :         :         :         g:         g: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name435754:&"
 
_user_specified_name435752:&"
 
_user_specified_name435750:&"
 
_user_specified_name435748:&"
 
_user_specified_name435746:&
"
 
_user_specified_name435744:&	"
 
_user_specified_name435742:&"
 
_user_specified_name435740:&"
 
_user_specified_name435738:&"
 
_user_specified_name435736:LH
'
_output_shapes
:         g

_user_specified_name1/1:LH
'
_output_shapes
:         g

_user_specified_name1/0:VR
'
_output_shapes
:         
'
_user_specified_name0/observation:OK
#
_output_shapes
:         
$
_user_specified_name
0/discount:MI
#
_output_shapes
:         
"
_user_specified_name
0/reward:P L
#
_output_shapes
:         
%
_user_specified_name0/step_type
ў
>
<__inference_signature_wrapper_function_with_signature_435846ў
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
GPU2*0J 8В *3
f.R,
*__inference_function_with_signature_435843*(
_construction_contextkEagerRuntime*
_input_shapes 
о
l
<__inference_signature_wrapper_function_with_signature_435828

batch_size
identity

identity_1░
PartitionedCallPartitionedCall
batch_size*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:         g:         g* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *3
f.R,
*__inference_function_with_signature_435820`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         gb

Identity_1IdentityPartitionedCall:output:1*
T0*'
_output_shapes
:         g"!

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
┌D
╧
"__inference__traced_restore_436650
file_prefix#
assignvariableop_variable:	 z
gassignvariableop_1_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_kernel:	Жt
eassignvariableop_2_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_bias:	Жt
`assignvariableop_3_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_kernel:
ЖЬ}
jassignvariableop_4_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_recurrent_kernel:	gЬm
^assignvariableop_5_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_bias:	Ьk
Yassignvariableop_6_actordistributionrnnnetwork_actordistributionrnnnetwork_dense_1_kernel:gde
Wassignvariableop_7_actordistributionrnnnetwork_actordistributionrnnnetwork_dense_1_bias:dd
Vassignvariableop_8_actordistributionrnnnetwork_normalprojectionnetwork_bias_layer_bias:v
dassignvariableop_9_actordistributionrnnnetwork_normalprojectionnetwork_means_projection_layer_kernel:dq
cassignvariableop_10_actordistributionrnnnetwork_normalprojectionnetwork_means_projection_layer_bias:
identity_12ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_2вAssignVariableOp_3вAssignVariableOp_4вAssignVariableOp_5вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9А
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ж
valueЬBЩB%train_step/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/0/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/1/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/2/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/3/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/4/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/5/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/6/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/7/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/8/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/9/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHИ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*+
value"B B B B B B B B B B B B B ┌
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*D
_output_shapes2
0::::::::::::*
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:м
AssignVariableOpAssignVariableOpassignvariableop_variableIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:■
AssignVariableOp_1AssignVariableOpgassignvariableop_1_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_kernelIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:№
AssignVariableOp_2AssignVariableOpeassignvariableop_2_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_biasIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:ў
AssignVariableOp_3AssignVariableOp`assignvariableop_3_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_kernelIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:Б
AssignVariableOp_4AssignVariableOpjassignvariableop_4_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_recurrent_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:ї
AssignVariableOp_5AssignVariableOp^assignvariableop_5_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:Ё
AssignVariableOp_6AssignVariableOpYassignvariableop_6_actordistributionrnnnetwork_actordistributionrnnnetwork_dense_1_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:ю
AssignVariableOp_7AssignVariableOpWassignvariableop_7_actordistributionrnnnetwork_actordistributionrnnnetwork_dense_1_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:э
AssignVariableOp_8AssignVariableOpVassignvariableop_8_actordistributionrnnnetwork_normalprojectionnetwork_bias_layer_biasIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:√
AssignVariableOp_9AssignVariableOpdassignvariableop_9_actordistributionrnnnetwork_normalprojectionnetwork_means_projection_layer_kernelIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:№
AssignVariableOp_10AssignVariableOpcassignvariableop_10_actordistributionrnnnetwork_normalprojectionnetwork_means_projection_layer_biasIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 ┴
Identity_11Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_12IdentityIdentity_11:output:0^NoOp_1*
T0*
_output_shapes
: К
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_12Identity_12:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
: : : : : : : : : : : : 2*
AssignVariableOp_10AssignVariableOp_102(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:ok
i
_user_specified_nameQOActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/bias:q
m
k
_user_specified_nameSQActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/kernel:c	_
]
_user_specified_nameECActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/bias:d`
^
_user_specified_nameFDActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/bias:fb
`
_user_specified_nameHFActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/kernel:kg
e
_user_specified_nameMKActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/bias:ws
q
_user_specified_nameYWActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/recurrent_kernel:mi
g
_user_specified_nameOMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/kernel:rn
l
_user_specified_nameTRActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/bias:tp
n
_user_specified_nameVTActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/kernel:($
"
_user_specified_name
Variable:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
╓
,
*__inference_function_with_signature_435843ш
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
GPU2*0J 8В *$
fR
__inference_<lambda>_435081*(
_construction_contextkEagerRuntime*
_input_shapes 
┼q
А
__inference__traced_save_436608
file_prefix)
read_disablecopyonread_variable:	 А
mread_1_disablecopyonread_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_kernel:	Жz
kread_2_disablecopyonread_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_bias:	Жz
fread_3_disablecopyonread_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_kernel:
ЖЬГ
pread_4_disablecopyonread_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_recurrent_kernel:	gЬs
dread_5_disablecopyonread_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_bias:	Ьq
_read_6_disablecopyonread_actordistributionrnnnetwork_actordistributionrnnnetwork_dense_1_kernel:gdk
]read_7_disablecopyonread_actordistributionrnnnetwork_actordistributionrnnnetwork_dense_1_bias:dj
\read_8_disablecopyonread_actordistributionrnnnetwork_normalprojectionnetwork_bias_layer_bias:|
jread_9_disablecopyonread_actordistributionrnnnetwork_normalprojectionnetwork_means_projection_layer_kernel:dw
iread_10_disablecopyonread_actordistributionrnnnetwork_normalprojectionnetwork_means_projection_layer_bias:
savev2_const
identity_23ИвMergeV2CheckpointsвRead/DisableCopyOnReadвRead/ReadVariableOpвRead_1/DisableCopyOnReadвRead_1/ReadVariableOpвRead_10/DisableCopyOnReadвRead_10/ReadVariableOpвRead_2/DisableCopyOnReadвRead_2/ReadVariableOpвRead_3/DisableCopyOnReadвRead_3/ReadVariableOpвRead_4/DisableCopyOnReadвRead_4/ReadVariableOpвRead_5/DisableCopyOnReadвRead_5/ReadVariableOpвRead_6/DisableCopyOnReadвRead_6/ReadVariableOpвRead_7/DisableCopyOnReadвRead_7/ReadVariableOpвRead_8/DisableCopyOnReadвRead_8/ReadVariableOpвRead_9/DisableCopyOnReadвRead_9/ReadVariableOpw
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
_temp/partБ
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
value	B : У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: q
Read/DisableCopyOnReadDisableCopyOnReadread_disablecopyonread_variable"/device:CPU:0*
_output_shapes
 У
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
: ┴
Read_1/DisableCopyOnReadDisableCopyOnReadmread_1_disablecopyonread_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_kernel"/device:CPU:0*
_output_shapes
 ю
Read_1/ReadVariableOpReadVariableOpmread_1_disablecopyonread_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_kernel^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	Ж*
dtype0n

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	Жd

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:	Ж┐
Read_2/DisableCopyOnReadDisableCopyOnReadkread_2_disablecopyonread_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_bias"/device:CPU:0*
_output_shapes
 ш
Read_2/ReadVariableOpReadVariableOpkread_2_disablecopyonread_actordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_bias^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:Ж*
dtype0j

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Ж`

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes	
:Ж║
Read_3/DisableCopyOnReadDisableCopyOnReadfread_3_disablecopyonread_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_kernel"/device:CPU:0*
_output_shapes
 ш
Read_3/ReadVariableOpReadVariableOpfread_3_disablecopyonread_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_kernel^Read_3/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
ЖЬ*
dtype0o

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
ЖЬe

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0* 
_output_shapes
:
ЖЬ─
Read_4/DisableCopyOnReadDisableCopyOnReadpread_4_disablecopyonread_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_recurrent_kernel"/device:CPU:0*
_output_shapes
 ё
Read_4/ReadVariableOpReadVariableOppread_4_disablecopyonread_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_recurrent_kernel^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	gЬ*
dtype0n

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	gЬd

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes
:	gЬ╕
Read_5/DisableCopyOnReadDisableCopyOnReaddread_5_disablecopyonread_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_bias"/device:CPU:0*
_output_shapes
 с
Read_5/ReadVariableOpReadVariableOpdread_5_disablecopyonread_actordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:Ь*
dtype0k
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:Ьb
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes	
:Ь│
Read_6/DisableCopyOnReadDisableCopyOnRead_read_6_disablecopyonread_actordistributionrnnnetwork_actordistributionrnnnetwork_dense_1_kernel"/device:CPU:0*
_output_shapes
 ▀
Read_6/ReadVariableOpReadVariableOp_read_6_disablecopyonread_actordistributionrnnnetwork_actordistributionrnnnetwork_dense_1_kernel^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:gd*
dtype0n
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:gde
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes

:gd▒
Read_7/DisableCopyOnReadDisableCopyOnRead]read_7_disablecopyonread_actordistributionrnnnetwork_actordistributionrnnnetwork_dense_1_bias"/device:CPU:0*
_output_shapes
 ┘
Read_7/ReadVariableOpReadVariableOp]read_7_disablecopyonread_actordistributionrnnnetwork_actordistributionrnnnetwork_dense_1_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:d*
dtype0j
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:da
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
:d░
Read_8/DisableCopyOnReadDisableCopyOnRead\read_8_disablecopyonread_actordistributionrnnnetwork_normalprojectionnetwork_bias_layer_bias"/device:CPU:0*
_output_shapes
 ╪
Read_8/ReadVariableOpReadVariableOp\read_8_disablecopyonread_actordistributionrnnnetwork_normalprojectionnetwork_bias_layer_bias^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0j
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes
:╛
Read_9/DisableCopyOnReadDisableCopyOnReadjread_9_disablecopyonread_actordistributionrnnnetwork_normalprojectionnetwork_means_projection_layer_kernel"/device:CPU:0*
_output_shapes
 ъ
Read_9/ReadVariableOpReadVariableOpjread_9_disablecopyonread_actordistributionrnnnetwork_normalprojectionnetwork_means_projection_layer_kernel^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:d*
dtype0n
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:de
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes

:d╛
Read_10/DisableCopyOnReadDisableCopyOnReadiread_10_disablecopyonread_actordistributionrnnnetwork_normalprojectionnetwork_means_projection_layer_bias"/device:CPU:0*
_output_shapes
 ч
Read_10/ReadVariableOpReadVariableOpiread_10_disablecopyonread_actordistributionrnnnetwork_normalprojectionnetwork_means_projection_layer_bias^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes
:¤
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ж
valueЬBЩB%train_step/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/0/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/1/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/2/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/3/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/4/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/5/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/6/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/7/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/8/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/9/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЕ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*+
value"B B B B B B B B B B B B B ╨
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtypes
2	Р
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:│
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_22Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_23IdentityIdentity_22:output:0^NoOp*
T0*
_output_shapes
: ц
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "#
identity_23Identity_23:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
: : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp24
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
Read_9/ReadVariableOpRead_9/ReadVariableOp:=9

_output_shapes
: 

_user_specified_nameConst:ok
i
_user_specified_nameQOActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/bias:q
m
k
_user_specified_nameSQActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/kernel:c	_
]
_user_specified_nameECActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/bias:d`
^
_user_specified_nameFDActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/bias:fb
`
_user_specified_nameHFActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/kernel:kg
e
_user_specified_nameMKActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/bias:ws
q
_user_specified_nameYWActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/recurrent_kernel:mi
g
_user_specified_nameOMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/kernel:rn
l
_user_specified_nameTRActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/bias:tp
n
_user_specified_nameVTActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/kernel:($
"
_user_specified_name
Variable:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ц
Z
*__inference_function_with_signature_435820

batch_size
identity

identity_1к
PartitionedCallPartitionedCall
batch_size*
Tin
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:         g:         g* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *-
f(R&
$__inference_get_initial_state_435815`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         gb

Identity_1IdentityPartitionedCall:output:1*
T0*'
_output_shapes
:         g"!

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
ЖЬ
╓
__inference_action_436075
	step_type

reward
discount
observation
unknown
	unknown_0
lactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_matmul_readvariableop_resource:	Ж|
mactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_biasadd_readvariableop_resource:	ЖГ
oactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_matmul_readvariableop_resource:
ЖЬД
qactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_matmul_1_readvariableop_resource:	gЬ
pactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_biasadd_readvariableop_resource:	Ьs
aactordistributionrnnnetwork_actordistributionrnnnetwork_dense_1_tensordot_readvariableop_resource:gdm
_actordistributionrnnnetwork_actordistributionrnnnetwork_dense_1_biasadd_readvariableop_resource:d{
iactordistributionrnnnetwork_normalprojectionnetwork_means_projection_layer_matmul_readvariableop_resource:dx
jactordistributionrnnnetwork_normalprojectionnetwork_means_projection_layer_biasadd_readvariableop_resource:l
^actordistributionrnnnetwork_normalprojectionnetwork_bias_layer_biasadd_readvariableop_resource:
identity

identity_1

identity_2ИвdActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpвcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpвVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/BiasAdd/ReadVariableOpвXActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/ReadVariableOpвgActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOpвfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOpвhActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOpвUActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOpвaActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOpв`ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOpK
ShapeShapediscount*
T0*
_output_shapes
::э╧]
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
valueB:╤
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
valueB:gM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Б
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
:         g[
shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB:gO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : З
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
:         gI
Equal/yConst*
_output_shapes
: *
dtype0*
value	B : Y
EqualEqual	step_typeEqual/y:output:0*
T0*#
_output_shapes
:         F
RankConst*
_output_shapes
: *
dtype0*
value	B :╛
PartitionedCallPartitionedCallzeros:output:0unknownRank:output:0	Equal:z:0*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         g* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *+
f&R$
"__inference_per_field_where_435891─
PartitionedCall_1PartitionedCallzeros_1:output:0	unknown_0Rank:output:0	Equal:z:0*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         g* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *+
f&R$
"__inference_per_field_where_435891И
FActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :ф
BActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims
ExpandDimsobservationOActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims/dim:output:0*
T0*+
_output_shapes
:         К
HActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :т
DActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1
ExpandDims	step_typeQActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:         Ї
[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/ShapeShapeKActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	:э╨┤
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"       ╒
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/ReshapeReshapeKActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims:output:0lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape/shape:output:0*
T0*'
_output_shapes
:         ж
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ▄
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten/ReshapeReshapefActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape:output:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten/Const:output:0*
T0*'
_output_shapes
:         С
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpReadVariableOplactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_matmul_readvariableop_resource*
_output_shapes
:	Ж*
dtype0р
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMulMatMul`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten/Reshape:output:0kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ЖП
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpReadVariableOpmactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_biasadd_readvariableop_resource*
_output_shapes	
:Ж*
dtype0с
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAddBiasAdd^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul:product:0lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Жэ
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/ReluRelu^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd:output:0*
T0*(
_output_shapes
:         Ж╡
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ╖
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:╖
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Е
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_sliceStridedSlicedActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Shape:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack:output:0vActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_1:output:0vActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_maskЛ
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/ShapeShape`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/Relu:activations:0*
T0*
_output_shapes
:*
out_type0	:э╨╖
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:╣
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ╣
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Н
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1StridedSlicefActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/Shape:output:0vActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack:output:0xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_1:output:0xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
end_maskе
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ш
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concatConcatV2nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice:output:0pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1:output:0lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat/axis:output:0*
N*
T0	*
_output_shapes
:·
_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/ReshapeReshape`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/Relu:activations:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat:output:0*
Tshape0	*
T0*,
_output_shapes
:         ЖА
>ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask/yConst*
_output_shapes
: *
dtype0*
value	B : П
<ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/maskEqualMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1:output:0GActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask/y:output:0*
T0*'
_output_shapes
:         Н
KActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/RankConst*
_output_shapes
: *
dtype0*
value	B :Ф
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/startConst*
_output_shapes
: *
dtype0*
value	B :Ф
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :Б
LActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/rangeRange[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/start:output:0TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Rank:output:0[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/delta:output:0*
_output_shapes
:з
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB"       Ф
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ь
MActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concatConcatV2_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/values_0:output:0UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range:output:0[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/axis:output:0*
N*
T0*
_output_shapes
:╓
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose	TransposehActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/Reshape:output:0VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat:output:0*
T0*,
_output_shapes
:         Ж▐
LActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ShapeShapeTActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose:y:0*
T0*
_output_shapes
::э╧д
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:ж
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:ж
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┤
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_sliceStridedSliceUActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Shape:output:0cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack:output:0eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_1:output:0eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskи
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       ╡
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1	Transpose@ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask:z:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1/perm:output:0*
T0
*'
_output_shapes
:         Ч
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :g╚
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packedPack]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice:output:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:Ч
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ┴
LActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zerosFill\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packed:output:0[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/Const:output:0*
T0*'
_output_shapes
:         gЩ
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :g╠
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packedPack]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice:output:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Щ
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ╟
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1Fill^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packed:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/Const:output:0*
T0*'
_output_shapes
:         g∙
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/SqueezeSqueezeTActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose:y:0*
T0*(
_output_shapes
:         Ж*
squeeze_dims
 °
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze_1SqueezeVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1:y:0*
T0
*#
_output_shapes
:         *
squeeze_dims
 ╒
MActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/SelectSelectYActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze_1:output:0UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros:output:0PartitionedCall:output:0*
T0*'
_output_shapes
:         g█
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Select_1SelectYActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze_1:output:0WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1:output:0PartitionedCall_1:output:0*
T0*'
_output_shapes
:         gШ
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOpReadVariableOpoactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_matmul_readvariableop_resource* 
_output_shapes
:
ЖЬ*
dtype0▌
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMulMatMulWActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze:output:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ЬЫ
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpqactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes
:	gЬ*
dtype0р
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1MatMulVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Select:output:0pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ь╪
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/addAddV2aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul:product:0cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:         ЬХ
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOpReadVariableOppactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:Ь*
dtype0с
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAddBiasAddXActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/add:z:0oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ьв
`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :й
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/splitSplitiActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split/split_dim:output:0aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:         g:         g:         g:         g*
	num_splitЎ
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/SigmoidSigmoid_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split:output:0*
T0*'
_output_shapes
:         g°
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_1Sigmoid_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split:output:1*
T0*'
_output_shapes
:         g╟
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mulMul^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_1:y:0XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Select_1:output:0*
T0*'
_output_shapes
:         gЁ
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/TanhTanh_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split:output:2*
T0*'
_output_shapes
:         g╚
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul_1Mul\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid:y:0YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:         g╟
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/add_1AddV2XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul:z:0ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:         g°
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_2Sigmoid_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split:output:3*
T0*'
_output_shapes
:         gэ
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Tanh_1TanhZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:         g╠
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul_2Mul^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_2:y:0[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:         gЧ
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :╤
QActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims
ExpandDimsZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul_2:z:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims/dim:output:0*
T0*+
_output_shapes
:         g·
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/ReadVariableOpReadVariableOpaactordistributionrnnnetwork_actordistributionrnnnetwork_dense_1_tensordot_readvariableop_resource*
_output_shapes

:gd*
dtype0Ш
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:Я
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ч
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/ShapeShapeZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims:output:0*
T0*
_output_shapes
::э╧Щ
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╗
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/GatherV2GatherV2XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/Shape:output:0WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/free:output:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Ы
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ┐
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/GatherV2_1GatherV2XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/Shape:output:0WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/axes:output:0bActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Щ
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: о
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/ProdProd[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/GatherV2:output:0XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: Ы
QActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ┤
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/Prod_1Prod]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/GatherV2_1:output:0ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: Ч
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ь
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/concatConcatV2WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/free:output:0WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/axes:output:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:╣
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/stackPackWActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/Prod:output:0YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:═
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/transpose	TransposeZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims:output:0YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:         g╩
QActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/ReshapeReshapeWActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/transpose:y:0XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  ╩
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/MatMulMatMulZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/Reshape:output:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         dЫ
QActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:dЩ
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : з
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/concat_1ConcatV2[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/GatherV2:output:0ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/Const_2:output:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:├
IActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/TensordotReshapeZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/MatMul:product:0[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         dЄ
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/BiasAdd/ReadVariableOpReadVariableOp_actordistributionrnnnetwork_actordistributionrnnnetwork_dense_1_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0╝
GActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/BiasAddBiasAddRActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot:output:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         d╘
DActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/ReluReluPActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/BiasAdd:output:0*
T0*+
_output_shapes
:         dч
?ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/SqueezeSqueezeRActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Relu:activations:0*
T0*'
_output_shapes
:         d*
squeeze_dims
К
`ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOpReadVariableOpiactordistributionrnnnetwork_normalprojectionnetwork_means_projection_layer_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0┴
QActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMulMatMulHActorDistributionRnnNetwork/ActorDistributionRnnNetwork/Squeeze:output:0hActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         И
aActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOpReadVariableOpjactordistributionrnnnetwork_normalprojectionnetwork_means_projection_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0╫
RActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAddBiasAdd[ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul:product:0iActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Ф
AActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         Э
;ActorDistributionRnnNetwork/NormalProjectionNetwork/ReshapeReshape[ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd:output:0JActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape/shape:output:0*
T0*#
_output_shapes
:         ┤
8ActorDistributionRnnNetwork/NormalProjectionNetwork/TanhTanhDActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape:output:0*
T0*#
_output_shapes
:         ~
9ActorDistributionRnnNetwork/NormalProjectionNetwork/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?ю
7ActorDistributionRnnNetwork/NormalProjectionNetwork/mulMulBActorDistributionRnnNetwork/NormalProjectionNetwork/mul/x:output:0<ActorDistributionRnnNetwork/NormalProjectionNetwork/Tanh:y:0*
T0*#
_output_shapes
:         ~
9ActorDistributionRnnNetwork/NormalProjectionNetwork/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?я
7ActorDistributionRnnNetwork/NormalProjectionNetwork/addAddV2BActorDistributionRnnNetwork/NormalProjectionNetwork/add/x:output:0;ActorDistributionRnnNetwork/NormalProjectionNetwork/mul:z:0*
T0*#
_output_shapes
:         ╢
>ActorDistributionRnnNetwork/NormalProjectionNetwork/zeros_like	ZerosLike;ActorDistributionRnnNetwork/NormalProjectionNetwork/add:z:0*
T0*#
_output_shapes
:         Ш
MActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
         е
IActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/ExpandDims
ExpandDimsBActorDistributionRnnNetwork/NormalProjectionNetwork/zeros_like:y:0VActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/ExpandDims/dim:output:0*
T0*'
_output_shapes
:         Ё
UActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOpReadVariableOp^actordistributionrnnnetwork_normalprojectionnetwork_bias_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0╢
FActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAddBiasAddRActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/ExpandDims:output:0]ActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         г
RActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        е
TActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       е
TActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ░
LActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/strided_sliceStridedSliceOActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd:output:0[ActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/strided_slice/stack:output:0]ActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/strided_slice/stack_1:output:0]ActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/strided_slice/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *
ellipsis_mask*
shrink_axis_maskЦ
CActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         Ы
=ActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape_1ReshapeUActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/strided_slice:output:0LActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape_1/shape:output:0*
T0*#
_output_shapes
:         ╛
<ActorDistributionRnnNetwork/NormalProjectionNetwork/SoftplusSoftplusFActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape_1:output:0*
T0*#
_output_shapes
:         а
]Normal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB ╧
VNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/ShapeShape;ActorDistributionRnnNetwork/NormalProjectionNetwork/add:z:0*
T0*
_output_shapes
::э╧Ш
VNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : о
dNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ░
fNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:░
fNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ф
^Normal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/strided_sliceStridedSlice_Normal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/Shape:output:0mNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/strided_slice/stack:output:0oNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/strided_slice/stack_1:output:0oNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskр
XNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/Shape_1ShapeJActorDistributionRnnNetwork/NormalProjectionNetwork/Softplus:activations:0*
T0*
_output_shapes
::э╧Ъ
XNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/Const_1Const*
_output_shapes
: *
dtype0*
value	B : ░
fNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ▓
hNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:▓
hNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ю
`Normal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/strided_slice_1StridedSliceaNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/Shape_1:output:0oNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/strided_slice_1/stack:output:0qNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/strided_slice_1/stack_1:output:0qNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskд
aNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB ж
cNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB т
^Normal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/BroadcastArgsBroadcastArgslNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/BroadcastArgs/s0_1:output:0gNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/strided_slice:output:0*
_output_shapes
:▌
`Normal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/BroadcastArgs_1BroadcastArgscNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/BroadcastArgs:r0:0iNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/strided_slice_1:output:0*
_output_shapes
:к
`Normal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:Ю
\Normal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ╩
WNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/concatConcatV2iNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/concat/values_0:output:0eNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/BroadcastArgs_1:r0:0eNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:п
jNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    ▒
lNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  А?│
zNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/normal/random_normal/RandomStandardNormalRandomStandardNormal`Normal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/concat:output:0*
T0*'
_output_shapes
:         *
dtype0Я
iNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/normal/random_normal/mulMulГNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/normal/random_normal/RandomStandardNormal:output:0uNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/normal/random_normal/stddev:output:0*
T0*'
_output_shapes
:         Д
eNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/normal/random_normalAddV2mNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/normal/random_normal/mul:z:0sNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/normal/random_normal/mean:output:0*
T0*'
_output_shapes
:         ─
TNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/mulMuliNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/normal/random_normal:z:0JActorDistributionRnnNetwork/NormalProjectionNetwork/Softplus:activations:0*
T0*'
_output_shapes
:         ж
TNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/addAddV2XNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/mul:z:0;ActorDistributionRnnNetwork/NormalProjectionNetwork/add:z:0*
T0*'
_output_shapes
:         б
^Normal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/Shape_2/ConstConst*
_output_shapes
: *
dtype0*
valueB в
XNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/Shape_2Const*
_output_shapes
:*
dtype0*
valueB: ю
XNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/Shape_3ShapeXNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/add:z:0*
T0*
_output_shapes
::э╧░
fNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:▓
hNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ▓
hNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ь
`Normal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/strided_slice_2StridedSliceaNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/Shape_3:output:0oNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/strided_slice_2/stack:output:0qNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/strided_slice_2/stack_1:output:0qNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskа
^Normal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ╧
YNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/concat_1ConcatV2fNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/sample_shape:output:0iNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/strided_slice_2:output:0gNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:╧
XNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/ReshapeReshapeXNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/add:z:0bNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/concat_1:output:0*
T0*#
_output_shapes
:         \
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╙
clip_by_value/MinimumMinimumaNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0*#
_output_shapes
:         T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    {
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*#
_output_shapes
:         \
IdentityIdentityclip_by_value:z:0^NoOp*
T0*#
_output_shapes
:         л

Identity_1IdentityZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul_2:z:0^NoOp*
T0*'
_output_shapes
:         gл

Identity_2IdentityZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/add_1:z:0^NoOp*
T0*'
_output_shapes
:         gА
NoOpNoOpe^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpd^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpW^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/BiasAdd/ReadVariableOpY^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/ReadVariableOph^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOpi^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOpV^ActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOpb^ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOpa^ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Н
_input_shapes|
z:         :         :         :         :         g:         g: : : : : : : : : : 2╠
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpdActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp2╩
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp2░
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/BiasAdd/ReadVariableOpVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/BiasAdd/ReadVariableOp2┤
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/ReadVariableOpXActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/ReadVariableOp2╥
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOpgActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOp2╨
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOpfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOp2╘
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOphActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOp2о
UActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOpUActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOp2╞
aActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOpaActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOp2─
`ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp`ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp:($
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
:         g

_user_specified_name1:JF
'
_output_shapes
:         g

_user_specified_name0:TP
'
_output_shapes
:         
%
_user_specified_nameobservation:MI
#
_output_shapes
:         
"
_user_specified_name
discount:KG
#
_output_shapes
:         
 
_user_specified_namereward:N J
#
_output_shapes
:         
#
_user_specified_name	step_type
Ў
|
<__inference_signature_wrapper_function_with_signature_435841
unknown:	 
identity	ИвStatefulPartitionedCallЫ
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
GPU2*0J 8В *3
f.R,
*__inference_function_with_signature_435834^
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
 
_user_specified_name435837
√
b
__inference_<lambda>_435079!
readvariableop_resource:	 
identity	ИвReadVariableOp^
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
╒
j
*__inference_function_with_signature_435834
unknown:	 
identity	ИвStatefulPartitionedCallМ
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
GPU2*0J 8В *$
fR
__inference_<lambda>_435079^
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
 
_user_specified_name435830
╩
_
"__inference_per_field_where_435551
t
f
sub_rank
shape_equal

identityY
assert_rank_at_least/ShapeShapet*
T0*
_output_shapes
::э╧a
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
::э╧e
ones/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         b
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
:         M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ~
concatConcatV2Shape:output:0ones:output:0concat/axis:output:0*
N*
T0*#
_output_shapes
:         S
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
7:         g:         g: :         :JF
#
_output_shapes
:         

_user_specified_nameEqual:<8

_output_shapes
: 

_user_specified_nameRank:JF
'
_output_shapes
:         g

_user_specified_namef:J F
'
_output_shapes
:         g

_user_specified_namet
╩
_
"__inference_per_field_where_436120
t
f
sub_rank
shape_equal

identityY
assert_rank_at_least/ShapeShapet*
T0*
_output_shapes
::э╧a
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
::э╧e
ones/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         b
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
:         M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ~
concatConcatV2Shape:output:0ones:output:0concat/axis:output:0*
N*
T0*#
_output_shapes
:         S
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
7:         g:         g: :         :JF
#
_output_shapes
:         

_user_specified_nameEqual:<8

_output_shapes
: 

_user_specified_nameRank:JF
'
_output_shapes
:         g

_user_specified_namef:J F
'
_output_shapes
:         g

_user_specified_namet
¤
T
$__inference_get_initial_state_436509

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
valueB:gM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Б
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
:         g[
shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB:gO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : З
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
:         gV
IdentityIdentityzeros:output:0*
T0*'
_output_shapes
:         gZ

Identity_1Identityzeros_1:output:0*
T0*'
_output_shapes
:         g"!

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
░Э
К
__inference_action_436304
time_step_step_type
time_step_reward
time_step_discount
time_step_observation
policy_state_0
policy_state_1
lactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_matmul_readvariableop_resource:	Ж|
mactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_biasadd_readvariableop_resource:	ЖГ
oactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_matmul_readvariableop_resource:
ЖЬД
qactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_matmul_1_readvariableop_resource:	gЬ
pactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_biasadd_readvariableop_resource:	Ьs
aactordistributionrnnnetwork_actordistributionrnnnetwork_dense_1_tensordot_readvariableop_resource:gdm
_actordistributionrnnnetwork_actordistributionrnnnetwork_dense_1_biasadd_readvariableop_resource:d{
iactordistributionrnnnetwork_normalprojectionnetwork_means_projection_layer_matmul_readvariableop_resource:dx
jactordistributionrnnnetwork_normalprojectionnetwork_means_projection_layer_biasadd_readvariableop_resource:l
^actordistributionrnnnetwork_normalprojectionnetwork_bias_layer_biasadd_readvariableop_resource:
identity

identity_1

identity_2ИвdActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpвcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpвVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/BiasAdd/ReadVariableOpвXActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/ReadVariableOpвgActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOpвfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOpвhActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOpвUActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOpвaActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOpв`ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOpU
ShapeShapetime_step_discount*
T0*
_output_shapes
::э╧]
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
valueB:╤
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
valueB:gM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Б
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
:         g[
shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB:gO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : З
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
:         gI
Equal/yConst*
_output_shapes
: *
dtype0*
value	B : c
EqualEqualtime_step_step_typeEqual/y:output:0*
T0*#
_output_shapes
:         F
RankConst*
_output_shapes
: *
dtype0*
value	B :┼
PartitionedCallPartitionedCallzeros:output:0policy_state_0Rank:output:0	Equal:z:0*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         g* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *+
f&R$
"__inference_per_field_where_436120╔
PartitionedCall_1PartitionedCallzeros_1:output:0policy_state_1Rank:output:0	Equal:z:0*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         g* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *+
f&R$
"__inference_per_field_where_436120И
FActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :ю
BActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims
ExpandDimstime_step_observationOActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims/dim:output:0*
T0*+
_output_shapes
:         К
HActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :ь
DActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1
ExpandDimstime_step_step_typeQActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:         Ї
[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/ShapeShapeKActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	:э╨┤
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"       ╒
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/ReshapeReshapeKActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims:output:0lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape/shape:output:0*
T0*'
_output_shapes
:         ж
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ▄
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten/ReshapeReshapefActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape:output:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten/Const:output:0*
T0*'
_output_shapes
:         С
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpReadVariableOplactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_matmul_readvariableop_resource*
_output_shapes
:	Ж*
dtype0р
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMulMatMul`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten/Reshape:output:0kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ЖП
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpReadVariableOpmactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_biasadd_readvariableop_resource*
_output_shapes	
:Ж*
dtype0с
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAddBiasAdd^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul:product:0lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Жэ
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/ReluRelu^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd:output:0*
T0*(
_output_shapes
:         Ж╡
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ╖
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:╖
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Е
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_sliceStridedSlicedActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Shape:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack:output:0vActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_1:output:0vActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_maskЛ
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/ShapeShape`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/Relu:activations:0*
T0*
_output_shapes
:*
out_type0	:э╨╖
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:╣
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ╣
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Н
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1StridedSlicefActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/Shape:output:0vActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack:output:0xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_1:output:0xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
end_maskе
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ш
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concatConcatV2nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice:output:0pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1:output:0lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat/axis:output:0*
N*
T0	*
_output_shapes
:·
_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/ReshapeReshape`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/Relu:activations:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat:output:0*
Tshape0	*
T0*,
_output_shapes
:         ЖА
>ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask/yConst*
_output_shapes
: *
dtype0*
value	B : П
<ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/maskEqualMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1:output:0GActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask/y:output:0*
T0*'
_output_shapes
:         Н
KActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/RankConst*
_output_shapes
: *
dtype0*
value	B :Ф
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/startConst*
_output_shapes
: *
dtype0*
value	B :Ф
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :Б
LActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/rangeRange[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/start:output:0TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Rank:output:0[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/delta:output:0*
_output_shapes
:з
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB"       Ф
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ь
MActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concatConcatV2_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/values_0:output:0UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range:output:0[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/axis:output:0*
N*
T0*
_output_shapes
:╓
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose	TransposehActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/Reshape:output:0VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat:output:0*
T0*,
_output_shapes
:         Ж▐
LActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ShapeShapeTActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose:y:0*
T0*
_output_shapes
::э╧д
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:ж
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:ж
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┤
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_sliceStridedSliceUActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Shape:output:0cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack:output:0eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_1:output:0eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskи
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       ╡
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1	Transpose@ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask:z:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1/perm:output:0*
T0
*'
_output_shapes
:         Ч
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :g╚
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packedPack]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice:output:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:Ч
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ┴
LActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zerosFill\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packed:output:0[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/Const:output:0*
T0*'
_output_shapes
:         gЩ
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :g╠
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packedPack]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice:output:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Щ
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ╟
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1Fill^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packed:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/Const:output:0*
T0*'
_output_shapes
:         g∙
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/SqueezeSqueezeTActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose:y:0*
T0*(
_output_shapes
:         Ж*
squeeze_dims
 °
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze_1SqueezeVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1:y:0*
T0
*#
_output_shapes
:         *
squeeze_dims
 ╒
MActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/SelectSelectYActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze_1:output:0UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros:output:0PartitionedCall:output:0*
T0*'
_output_shapes
:         g█
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Select_1SelectYActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze_1:output:0WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1:output:0PartitionedCall_1:output:0*
T0*'
_output_shapes
:         gШ
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOpReadVariableOpoactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_matmul_readvariableop_resource* 
_output_shapes
:
ЖЬ*
dtype0▌
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMulMatMulWActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze:output:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ЬЫ
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpqactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes
:	gЬ*
dtype0р
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1MatMulVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Select:output:0pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ь╪
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/addAddV2aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul:product:0cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:         ЬХ
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOpReadVariableOppactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:Ь*
dtype0с
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAddBiasAddXActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/add:z:0oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ьв
`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :й
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/splitSplitiActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split/split_dim:output:0aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:         g:         g:         g:         g*
	num_splitЎ
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/SigmoidSigmoid_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split:output:0*
T0*'
_output_shapes
:         g°
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_1Sigmoid_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split:output:1*
T0*'
_output_shapes
:         g╟
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mulMul^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_1:y:0XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Select_1:output:0*
T0*'
_output_shapes
:         gЁ
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/TanhTanh_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split:output:2*
T0*'
_output_shapes
:         g╚
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul_1Mul\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid:y:0YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:         g╟
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/add_1AddV2XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul:z:0ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:         g°
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_2Sigmoid_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split:output:3*
T0*'
_output_shapes
:         gэ
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Tanh_1TanhZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:         g╠
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul_2Mul^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_2:y:0[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:         gЧ
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :╤
QActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims
ExpandDimsZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul_2:z:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims/dim:output:0*
T0*+
_output_shapes
:         g·
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/ReadVariableOpReadVariableOpaactordistributionrnnnetwork_actordistributionrnnnetwork_dense_1_tensordot_readvariableop_resource*
_output_shapes

:gd*
dtype0Ш
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:Я
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ч
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/ShapeShapeZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims:output:0*
T0*
_output_shapes
::э╧Щ
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╗
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/GatherV2GatherV2XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/Shape:output:0WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/free:output:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Ы
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ┐
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/GatherV2_1GatherV2XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/Shape:output:0WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/axes:output:0bActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Щ
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: о
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/ProdProd[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/GatherV2:output:0XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: Ы
QActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ┤
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/Prod_1Prod]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/GatherV2_1:output:0ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: Ч
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ь
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/concatConcatV2WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/free:output:0WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/axes:output:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:╣
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/stackPackWActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/Prod:output:0YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:═
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/transpose	TransposeZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims:output:0YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:         g╩
QActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/ReshapeReshapeWActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/transpose:y:0XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  ╩
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/MatMulMatMulZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/Reshape:output:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         dЫ
QActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:dЩ
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : з
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/concat_1ConcatV2[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/GatherV2:output:0ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/Const_2:output:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:├
IActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/TensordotReshapeZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/MatMul:product:0[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         dЄ
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/BiasAdd/ReadVariableOpReadVariableOp_actordistributionrnnnetwork_actordistributionrnnnetwork_dense_1_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0╝
GActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/BiasAddBiasAddRActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot:output:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         d╘
DActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/ReluReluPActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/BiasAdd:output:0*
T0*+
_output_shapes
:         dч
?ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/SqueezeSqueezeRActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Relu:activations:0*
T0*'
_output_shapes
:         d*
squeeze_dims
К
`ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOpReadVariableOpiactordistributionrnnnetwork_normalprojectionnetwork_means_projection_layer_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0┴
QActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMulMatMulHActorDistributionRnnNetwork/ActorDistributionRnnNetwork/Squeeze:output:0hActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         И
aActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOpReadVariableOpjactordistributionrnnnetwork_normalprojectionnetwork_means_projection_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0╫
RActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAddBiasAdd[ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul:product:0iActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Ф
AActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         Э
;ActorDistributionRnnNetwork/NormalProjectionNetwork/ReshapeReshape[ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd:output:0JActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape/shape:output:0*
T0*#
_output_shapes
:         ┤
8ActorDistributionRnnNetwork/NormalProjectionNetwork/TanhTanhDActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape:output:0*
T0*#
_output_shapes
:         ~
9ActorDistributionRnnNetwork/NormalProjectionNetwork/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?ю
7ActorDistributionRnnNetwork/NormalProjectionNetwork/mulMulBActorDistributionRnnNetwork/NormalProjectionNetwork/mul/x:output:0<ActorDistributionRnnNetwork/NormalProjectionNetwork/Tanh:y:0*
T0*#
_output_shapes
:         ~
9ActorDistributionRnnNetwork/NormalProjectionNetwork/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?я
7ActorDistributionRnnNetwork/NormalProjectionNetwork/addAddV2BActorDistributionRnnNetwork/NormalProjectionNetwork/add/x:output:0;ActorDistributionRnnNetwork/NormalProjectionNetwork/mul:z:0*
T0*#
_output_shapes
:         ╢
>ActorDistributionRnnNetwork/NormalProjectionNetwork/zeros_like	ZerosLike;ActorDistributionRnnNetwork/NormalProjectionNetwork/add:z:0*
T0*#
_output_shapes
:         Ш
MActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
         е
IActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/ExpandDims
ExpandDimsBActorDistributionRnnNetwork/NormalProjectionNetwork/zeros_like:y:0VActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/ExpandDims/dim:output:0*
T0*'
_output_shapes
:         Ё
UActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOpReadVariableOp^actordistributionrnnnetwork_normalprojectionnetwork_bias_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0╢
FActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAddBiasAddRActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/ExpandDims:output:0]ActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         г
RActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        е
TActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       е
TActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ░
LActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/strided_sliceStridedSliceOActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd:output:0[ActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/strided_slice/stack:output:0]ActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/strided_slice/stack_1:output:0]ActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/strided_slice/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *
ellipsis_mask*
shrink_axis_maskЦ
CActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         Ы
=ActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape_1ReshapeUActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/strided_slice:output:0LActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape_1/shape:output:0*
T0*#
_output_shapes
:         ╛
<ActorDistributionRnnNetwork/NormalProjectionNetwork/SoftplusSoftplusFActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape_1:output:0*
T0*#
_output_shapes
:         а
]Normal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB ╧
VNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/ShapeShape;ActorDistributionRnnNetwork/NormalProjectionNetwork/add:z:0*
T0*
_output_shapes
::э╧Ш
VNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : о
dNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ░
fNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:░
fNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ф
^Normal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/strided_sliceStridedSlice_Normal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/Shape:output:0mNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/strided_slice/stack:output:0oNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/strided_slice/stack_1:output:0oNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskр
XNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/Shape_1ShapeJActorDistributionRnnNetwork/NormalProjectionNetwork/Softplus:activations:0*
T0*
_output_shapes
::э╧Ъ
XNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/Const_1Const*
_output_shapes
: *
dtype0*
value	B : ░
fNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ▓
hNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:▓
hNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ю
`Normal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/strided_slice_1StridedSliceaNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/Shape_1:output:0oNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/strided_slice_1/stack:output:0qNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/strided_slice_1/stack_1:output:0qNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskд
aNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB ж
cNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB т
^Normal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/BroadcastArgsBroadcastArgslNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/BroadcastArgs/s0_1:output:0gNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/strided_slice:output:0*
_output_shapes
:▌
`Normal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/BroadcastArgs_1BroadcastArgscNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/BroadcastArgs:r0:0iNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/strided_slice_1:output:0*
_output_shapes
:к
`Normal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:Ю
\Normal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ╩
WNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/concatConcatV2iNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/concat/values_0:output:0eNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/BroadcastArgs_1:r0:0eNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:п
jNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    ▒
lNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  А?│
zNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/normal/random_normal/RandomStandardNormalRandomStandardNormal`Normal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/concat:output:0*
T0*'
_output_shapes
:         *
dtype0Я
iNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/normal/random_normal/mulMulГNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/normal/random_normal/RandomStandardNormal:output:0uNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/normal/random_normal/stddev:output:0*
T0*'
_output_shapes
:         Д
eNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/normal/random_normalAddV2mNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/normal/random_normal/mul:z:0sNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/normal/random_normal/mean:output:0*
T0*'
_output_shapes
:         ─
TNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/mulMuliNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/normal/random_normal:z:0JActorDistributionRnnNetwork/NormalProjectionNetwork/Softplus:activations:0*
T0*'
_output_shapes
:         ж
TNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/addAddV2XNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/mul:z:0;ActorDistributionRnnNetwork/NormalProjectionNetwork/add:z:0*
T0*'
_output_shapes
:         б
^Normal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/Shape_2/ConstConst*
_output_shapes
: *
dtype0*
valueB в
XNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/Shape_2Const*
_output_shapes
:*
dtype0*
valueB: ю
XNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/Shape_3ShapeXNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/add:z:0*
T0*
_output_shapes
::э╧░
fNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:▓
hNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ▓
hNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ь
`Normal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/strided_slice_2StridedSliceaNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/Shape_3:output:0oNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/strided_slice_2/stack:output:0qNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/strided_slice_2/stack_1:output:0qNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskа
^Normal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ╧
YNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/concat_1ConcatV2fNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/sample_shape:output:0iNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/strided_slice_2:output:0gNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:╧
XNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/ReshapeReshapeXNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/add:z:0bNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/concat_1:output:0*
T0*#
_output_shapes
:         \
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?╙
clip_by_value/MinimumMinimumaNormal_CONSTRUCTED_AT_ActorDistributionRnnNetwork_NormalProjectionNetwork/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0*#
_output_shapes
:         T
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    {
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*#
_output_shapes
:         \
IdentityIdentityclip_by_value:z:0^NoOp*
T0*#
_output_shapes
:         л

Identity_1IdentityZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul_2:z:0^NoOp*
T0*'
_output_shapes
:         gл

Identity_2IdentityZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/add_1:z:0^NoOp*
T0*'
_output_shapes
:         gА
NoOpNoOpe^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpd^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpW^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/BiasAdd/ReadVariableOpY^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/ReadVariableOph^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOpi^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOpV^ActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOpb^ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOpa^ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Н
_input_shapes|
z:         :         :         :         :         g:         g: : : : : : : : : : 2╠
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpdActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp2╩
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp2░
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/BiasAdd/ReadVariableOpVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/BiasAdd/ReadVariableOp2┤
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/ReadVariableOpXActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/ReadVariableOp2╥
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOpgActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOp2╨
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOpfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOp2╘
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOphActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOp2о
UActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOpUActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOp2╞
aActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOpaActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOp2─
`ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp`ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp:($
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
:         g
(
_user_specified_namepolicy_state_1:WS
'
_output_shapes
:         g
(
_user_specified_namepolicy_state_0:^Z
'
_output_shapes
:         
/
_user_specified_nametime_step_observation:WS
#
_output_shapes
:         
,
_user_specified_nametime_step_discount:UQ
#
_output_shapes
:         
*
_user_specified_nametime_step_reward:X T
#
_output_shapes
:         
-
_user_specified_nametime_step_step_type
Є╠
я
"__inference_distribution_fn_436493
	step_type

reward
discount
observation
unknown
	unknown_0
lactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_matmul_readvariableop_resource:	Ж|
mactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_biasadd_readvariableop_resource:	ЖГ
oactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_matmul_readvariableop_resource:
ЖЬД
qactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_matmul_1_readvariableop_resource:	gЬ
pactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_biasadd_readvariableop_resource:	Ьs
aactordistributionrnnnetwork_actordistributionrnnnetwork_dense_1_tensordot_readvariableop_resource:gdm
_actordistributionrnnnetwork_actordistributionrnnnetwork_dense_1_biasadd_readvariableop_resource:d{
iactordistributionrnnnetwork_normalprojectionnetwork_means_projection_layer_matmul_readvariableop_resource:dx
jactordistributionrnnnetwork_normalprojectionnetwork_means_projection_layer_biasadd_readvariableop_resource:l
^actordistributionrnnnetwork_normalprojectionnetwork_bias_layer_biasadd_readvariableop_resource:
identity

identity_1

identity_2

identity_3ИвdActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpвcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpвVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/BiasAdd/ReadVariableOpвXActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/ReadVariableOpвgActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOpвfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOpвhActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOpвUActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOpвaActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOpв`ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOpK
ShapeShapediscount*
T0*
_output_shapes
::э╧]
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
valueB:╤
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
valueB:gM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Б
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
:         g[
shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB:gO
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : З
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
:         gI
Equal/yConst*
_output_shapes
: *
dtype0*
value	B : Y
EqualEqual	step_typeEqual/y:output:0*
T0*#
_output_shapes
:         F
RankConst*
_output_shapes
: *
dtype0*
value	B :╛
PartitionedCallPartitionedCallzeros:output:0unknownRank:output:0	Equal:z:0*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         g* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *+
f&R$
"__inference_per_field_where_436349─
PartitionedCall_1PartitionedCallzeros_1:output:0	unknown_0Rank:output:0	Equal:z:0*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         g* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *+
f&R$
"__inference_per_field_where_436349И
FActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :ф
BActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims
ExpandDimsobservationOActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims/dim:output:0*
T0*+
_output_shapes
:         К
HActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :т
DActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1
ExpandDims	step_typeQActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1/dim:output:0*
T0*'
_output_shapes
:         Ї
[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/ShapeShapeKActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims:output:0*
T0*
_output_shapes
:*
out_type0	:э╨┤
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"       ╒
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/ReshapeReshapeKActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims:output:0lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape/shape:output:0*
T0*'
_output_shapes
:         ж
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ▄
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten/ReshapeReshapefActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Reshape:output:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten/Const:output:0*
T0*'
_output_shapes
:         С
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpReadVariableOplactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_matmul_readvariableop_resource*
_output_shapes
:	Ж*
dtype0р
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMulMatMul`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/flatten/Reshape:output:0kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ЖП
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpReadVariableOpmactordistributionrnnnetwork_actordistributionrnnnetwork_encodingnetwork_dense_biasadd_readvariableop_resource*
_output_shapes	
:Ж*
dtype0с
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAddBiasAdd^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul:product:0lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Жэ
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/ReluRelu^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd:output:0*
T0*(
_output_shapes
:         Ж╡
kActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ╖
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:╖
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Е
eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_sliceStridedSlicedActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_flatten/Shape:output:0tActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack:output:0vActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_1:output:0vActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_maskЛ
]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/ShapeShape`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/Relu:activations:0*
T0*
_output_shapes
:*
out_type0	:э╨╖
mActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:╣
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: ╣
oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Н
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1StridedSlicefActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/Shape:output:0vActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack:output:0xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_1:output:0xActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
end_maskе
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ш
^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concatConcatV2nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice:output:0pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/strided_slice_1:output:0lActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat/axis:output:0*
N*
T0	*
_output_shapes
:·
_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/ReshapeReshape`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/Relu:activations:0gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/concat:output:0*
Tshape0	*
T0*,
_output_shapes
:         ЖА
>ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask/yConst*
_output_shapes
: *
dtype0*
value	B : П
<ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/maskEqualMActorDistributionRnnNetwork/ActorDistributionRnnNetwork/ExpandDims_1:output:0GActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask/y:output:0*
T0*'
_output_shapes
:         Н
KActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/RankConst*
_output_shapes
: *
dtype0*
value	B :Ф
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/startConst*
_output_shapes
: *
dtype0*
value	B :Ф
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :Б
LActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/rangeRange[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/start:output:0TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Rank:output:0[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range/delta:output:0*
_output_shapes
:з
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB"       Ф
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ь
MActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concatConcatV2_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/values_0:output:0UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/range:output:0[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat/axis:output:0*
N*
T0*
_output_shapes
:╓
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose	TransposehActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/batch_unflatten/Reshape:output:0VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/concat:output:0*
T0*,
_output_shapes
:         Ж▐
LActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ShapeShapeTActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose:y:0*
T0*
_output_shapes
::э╧д
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:ж
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:ж
\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┤
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_sliceStridedSliceUActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Shape:output:0cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack:output:0eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_1:output:0eActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskи
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       ╡
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1	Transpose@ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/mask:z:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1/perm:output:0*
T0
*'
_output_shapes
:         Ч
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :g╚
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packedPack]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice:output:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:Ч
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ┴
LActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zerosFill\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/packed:output:0[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros/Const:output:0*
T0*'
_output_shapes
:         gЩ
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :g╠
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packedPack]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/strided_slice:output:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Щ
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ╟
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1Fill^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/packed:output:0]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1/Const:output:0*
T0*'
_output_shapes
:         g∙
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/SqueezeSqueezeTActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose:y:0*
T0*(
_output_shapes
:         Ж*
squeeze_dims
 °
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze_1SqueezeVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/transpose_1:y:0*
T0
*#
_output_shapes
:         *
squeeze_dims
 ╒
MActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/SelectSelectYActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze_1:output:0UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros:output:0PartitionedCall:output:0*
T0*'
_output_shapes
:         g█
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Select_1SelectYActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze_1:output:0WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/zeros_1:output:0PartitionedCall_1:output:0*
T0*'
_output_shapes
:         gШ
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOpReadVariableOpoactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_matmul_readvariableop_resource* 
_output_shapes
:
ЖЬ*
dtype0▌
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMulMatMulWActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Squeeze:output:0nActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ЬЫ
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOpReadVariableOpqactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_matmul_1_readvariableop_resource*
_output_shapes
:	gЬ*
dtype0р
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1MatMulVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Select:output:0pActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ь╪
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/addAddV2aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul:product:0cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1:product:0*
T0*(
_output_shapes
:         ЬХ
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOpReadVariableOppactordistributionrnnnetwork_actordistributionrnnnetwork_dynamic_unroll_lstm_cell_biasadd_readvariableop_resource*
_output_shapes	
:Ь*
dtype0с
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAddBiasAddXActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/add:z:0oActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ьв
`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :й
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/splitSplitiActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split/split_dim:output:0aActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd:output:0*
T0*`
_output_shapesN
L:         g:         g:         g:         g*
	num_splitЎ
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/SigmoidSigmoid_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split:output:0*
T0*'
_output_shapes
:         g°
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_1Sigmoid_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split:output:1*
T0*'
_output_shapes
:         g╟
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mulMul^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_1:y:0XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/Select_1:output:0*
T0*'
_output_shapes
:         gЁ
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/TanhTanh_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split:output:2*
T0*'
_output_shapes
:         g╚
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul_1Mul\ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid:y:0YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Tanh:y:0*
T0*'
_output_shapes
:         g╟
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/add_1AddV2XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul:z:0ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul_1:z:0*
T0*'
_output_shapes
:         g°
ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_2Sigmoid_ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/split:output:3*
T0*'
_output_shapes
:         gэ
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Tanh_1TanhZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/add_1:z:0*
T0*'
_output_shapes
:         g╠
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul_2Mul^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Sigmoid_2:y:0[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/Tanh_1:y:0*
T0*'
_output_shapes
:         gЧ
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :╤
QActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims
ExpandDimsZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul_2:z:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims/dim:output:0*
T0*+
_output_shapes
:         g·
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/ReadVariableOpReadVariableOpaactordistributionrnnnetwork_actordistributionrnnnetwork_dense_1_tensordot_readvariableop_resource*
_output_shapes

:gd*
dtype0Ш
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:Я
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ч
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/ShapeShapeZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims:output:0*
T0*
_output_shapes
::э╧Щ
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╗
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/GatherV2GatherV2XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/Shape:output:0WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/free:output:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Ы
YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ┐
TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/GatherV2_1GatherV2XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/Shape:output:0WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/axes:output:0bActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Щ
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: о
NActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/ProdProd[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/GatherV2:output:0XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: Ы
QActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ┤
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/Prod_1Prod]ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/GatherV2_1:output:0ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: Ч
UActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ь
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/concatConcatV2WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/free:output:0WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/axes:output:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:╣
OActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/stackPackWActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/Prod:output:0YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:═
SActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/transpose	TransposeZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/ExpandDims:output:0YActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:         g╩
QActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/ReshapeReshapeWActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/transpose:y:0XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  ╩
PActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/MatMulMatMulZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/Reshape:output:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         dЫ
QActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:dЩ
WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : з
RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/concat_1ConcatV2[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/GatherV2:output:0ZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/Const_2:output:0`ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:├
IActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/TensordotReshapeZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/MatMul:product:0[ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         dЄ
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/BiasAdd/ReadVariableOpReadVariableOp_actordistributionrnnnetwork_actordistributionrnnnetwork_dense_1_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0╝
GActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/BiasAddBiasAddRActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot:output:0^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         d╘
DActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/ReluReluPActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/BiasAdd:output:0*
T0*+
_output_shapes
:         dч
?ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/SqueezeSqueezeRActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Relu:activations:0*
T0*'
_output_shapes
:         d*
squeeze_dims
К
`ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOpReadVariableOpiactordistributionrnnnetwork_normalprojectionnetwork_means_projection_layer_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0┴
QActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMulMatMulHActorDistributionRnnNetwork/ActorDistributionRnnNetwork/Squeeze:output:0hActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         И
aActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOpReadVariableOpjactordistributionrnnnetwork_normalprojectionnetwork_means_projection_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0╫
RActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAddBiasAdd[ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul:product:0iActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Ф
AActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         Э
;ActorDistributionRnnNetwork/NormalProjectionNetwork/ReshapeReshape[ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd:output:0JActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape/shape:output:0*
T0*#
_output_shapes
:         ┤
8ActorDistributionRnnNetwork/NormalProjectionNetwork/TanhTanhDActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape:output:0*
T0*#
_output_shapes
:         ~
9ActorDistributionRnnNetwork/NormalProjectionNetwork/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?ю
7ActorDistributionRnnNetwork/NormalProjectionNetwork/mulMulBActorDistributionRnnNetwork/NormalProjectionNetwork/mul/x:output:0<ActorDistributionRnnNetwork/NormalProjectionNetwork/Tanh:y:0*
T0*#
_output_shapes
:         ~
9ActorDistributionRnnNetwork/NormalProjectionNetwork/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?я
7ActorDistributionRnnNetwork/NormalProjectionNetwork/addAddV2BActorDistributionRnnNetwork/NormalProjectionNetwork/add/x:output:0;ActorDistributionRnnNetwork/NormalProjectionNetwork/mul:z:0*
T0*#
_output_shapes
:         ╢
>ActorDistributionRnnNetwork/NormalProjectionNetwork/zeros_like	ZerosLike;ActorDistributionRnnNetwork/NormalProjectionNetwork/add:z:0*
T0*#
_output_shapes
:         Ш
MActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
         е
IActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/ExpandDims
ExpandDimsBActorDistributionRnnNetwork/NormalProjectionNetwork/zeros_like:y:0VActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/ExpandDims/dim:output:0*
T0*'
_output_shapes
:         Ё
UActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOpReadVariableOp^actordistributionrnnnetwork_normalprojectionnetwork_bias_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0╢
FActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAddBiasAddRActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/ExpandDims:output:0]ActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         г
RActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        е
TActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       е
TActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ░
LActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/strided_sliceStridedSliceOActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd:output:0[ActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/strided_slice/stack:output:0]ActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/strided_slice/stack_1:output:0]ActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/strided_slice/stack_2:output:0*
Index0*
T0*#
_output_shapes
:         *
ellipsis_mask*
shrink_axis_maskЦ
CActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         Ы
=ActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape_1ReshapeUActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/strided_slice:output:0LActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape_1/shape:output:0*
T0*#
_output_shapes
:         ╛
<ActorDistributionRnnNetwork/NormalProjectionNetwork/SoftplusSoftplusFActorDistributionRnnNetwork/NormalProjectionNetwork/Reshape_1:output:0*
T0*#
_output_shapes
:         Ж
IdentityIdentity;ActorDistributionRnnNetwork/NormalProjectionNetwork/add:z:0^NoOp*
T0*#
_output_shapes
:         Ч

Identity_1IdentityJActorDistributionRnnNetwork/NormalProjectionNetwork/Softplus:activations:0^NoOp*
T0*#
_output_shapes
:         л

Identity_2IdentityZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/mul_2:z:0^NoOp*
T0*'
_output_shapes
:         gл

Identity_3IdentityZActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/add_1:z:0^NoOp*
T0*'
_output_shapes
:         gА
NoOpNoOpe^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpd^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpW^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/BiasAdd/ReadVariableOpY^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/ReadVariableOph^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOpg^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOpi^ActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOpV^ActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOpb^ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOpa^ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Н
_input_shapes|
z:         :         :         :         :         g:         g: : : : : : : : : : 2╠
dActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpdActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp2╩
cActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpcActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp2░
VActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/BiasAdd/ReadVariableOpVActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/BiasAdd/ReadVariableOp2┤
XActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/ReadVariableOpXActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/Tensordot/ReadVariableOp2╥
gActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOpgActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/BiasAdd/ReadVariableOp2╨
fActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOpfActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul/ReadVariableOp2╘
hActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOphActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/lstm_cell/MatMul_1/ReadVariableOp2о
UActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOpUActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/BiasAdd/ReadVariableOp2╞
aActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOpaActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/BiasAdd/ReadVariableOp2─
`ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp`ActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/MatMul/ReadVariableOp:($
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
:         g

_user_specified_name1:JF
'
_output_shapes
:         g

_user_specified_name0:TP
'
_output_shapes
:         
%
_user_specified_nameobservation:MI
#
_output_shapes
:         
"
_user_specified_name
discount:KG
#
_output_shapes
:         
 
_user_specified_namereward:N J
#
_output_shapes
:         
#
_user_specified_name	step_type
╩
_
"__inference_per_field_where_436349
t
f
sub_rank
shape_equal

identityY
assert_rank_at_least/ShapeShapet*
T0*
_output_shapes
::э╧a
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
::э╧e
ones/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         b
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
:         M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ~
concatConcatV2Shape:output:0ones:output:0concat/axis:output:0*
N*
T0*#
_output_shapes
:         S
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
7:         g:         g: :         :JF
#
_output_shapes
:         

_user_specified_nameEqual:<8

_output_shapes
: 

_user_specified_nameRank:JF
'
_output_shapes
:         g

_user_specified_namef:J F
'
_output_shapes
:         g

_user_specified_namet
╩
_
"__inference_per_field_where_435891
t
f
sub_rank
shape_equal

identityY
assert_rank_at_least/ShapeShapet*
T0*
_output_shapes
::э╧a
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
::э╧e
ones/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
         b
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
:         M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ~
concatConcatV2Shape:output:0ones:output:0concat/axis:output:0*
N*
T0*#
_output_shapes
:         S
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
7:         g:         g: :         :JF
#
_output_shapes
:         

_user_specified_nameEqual:<8

_output_shapes
: 

_user_specified_nameRank:JF
'
_output_shapes
:         g

_user_specified_namef:J F
'
_output_shapes
:         g

_user_specified_namet
\

__inference_<lambda>_435081*(
_construction_contextkEagerRuntime*
_input_shapes "эL
saver_filename:0StatefulPartitionedCall_2:0StatefulPartitionedCall_38"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*С
actionЖ
4

0/discount&
action_0_discount:0         
>
0/observation-
action_0_observation:0         
0
0/reward$
action_0_reward:0         
6
0/step_type'
action_0_step_type:0         
*
1/0#
action_1_0:0         g
*
1/1#
action_1_1:0         g6
action,
StatefulPartitionedCall:0         ;
state/00
StatefulPartitionedCall:1         g;
state/10
StatefulPartitionedCall:2         gtensorflow/serving/predict*─
get_initial_stateо
2

batch_size$
get_initial_state_batch_size:0 -
0(
PartitionedCall:0         g-
1(
PartitionedCall:1         gtensorflow/serving/predict*,
get_metadatatensorflow/serving/predict*Z
get_train_stepH*
int64!
StatefulPartitionedCall_1:0	 tensorflow/serving/predict:┐з
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
g
0
1
2
3
4
5
6
7
8
9"
trackable_tuple_wrapper
c
_actor_network
_policy_state_spec
_policy_step_spec"
trackable_dict_wrapper
н
trace_0
trace_12Ў
__inference_action_436075
__inference_action_436304╜
╢▓▓
FullArgSpec0
args(Ъ%
j	time_step
jpolicy_state
jseed
varargs
 
varkw
 
defaultsв	
в 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 ztrace_0ztrace_1
я
trace_02╥
"__inference_distribution_fn_436493л
д▓а
FullArgSpec(
args Ъ
j	time_step
jpolicy_state
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 ztrace_0
т
trace_02┼
$__inference_get_initial_state_436509Ь
Х▓С
FullArgSpec
argsЪ
j
batch_size
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 ztrace_0
▓Bп
__inference_<lambda>_435081"П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в 
▓Bп
__inference_<lambda>_435079"П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в 
`

action
get_initial_state
get_train_step
 get_metadata"
signature_map
i:g	Ж 2TActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/kernel
c:aЖ 2RActorDistributionRnnNetwork/ActorDistributionRnnNetwork/EncodingNetwork/dense/bias
c:a
ЖЬ 2MActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/kernel
l:j	gЬ 2WActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/recurrent_kernel
\:ZЬ 2KActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dynamic_unroll/bias
Z:Xgd 2FActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/kernel
T:Rd 2DActorDistributionRnnNetwork/ActorDistributionRnnNetwork/dense_1/bias
S:Q 2CActorDistributionRnnNetwork/NormalProjectionNetwork/bias_layer/bias
e:cd 2QActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/kernel
_:] 2OActorDistributionRnnNetwork/NormalProjectionNetwork/means_projection_layer/bias
у
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses
'_state_spec
(_lstm_encoder
)_projection_networks"
_tf_keras_layer
 "
trackable_list_wrapper
3
	state
1"
trackable_tuple_wrapper
ЖBГ
__inference_action_436075	step_typerewarddiscountobservation01"│
м▓и
FullArgSpec0
args(Ъ%
j	time_step
jpolicy_state
jseed
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╚B┼
__inference_action_436304time_step_step_typetime_step_rewardtime_step_discounttime_step_observationpolicy_state_0policy_state_1"│
м▓и
FullArgSpec0
args(Ъ%
j	time_step
jpolicy_state
jseed
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЗBД
"__inference_distribution_fn_436493	step_typerewarddiscountobservation01"л
д▓а
FullArgSpec(
args Ъ
j	time_step
jpolicy_state
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╓B╙
$__inference_get_initial_state_436509
batch_size"Ь
Х▓С
FullArgSpec
argsЪ
j
batch_size
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
юBы
<__inference_signature_wrapper_function_with_signature_435797
0/discount0/observation0/reward0/step_type1/01/1"ю
ч▓у
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 q

kwonlyargscЪ`
jarg_0_discount
jarg_0_observation
jarg_0_reward
jarg_0_step_type
	jarg_1_0
	jarg_1_1
kwonlydefaults
 
annotationsк *
 
юBы
<__inference_signature_wrapper_function_with_signature_435828
batch_size"Ь
Х▓С
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ
j
batch_size
kwonlydefaults
 
annotationsк *
 
╙B╨
<__inference_signature_wrapper_function_with_signature_435841"П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в 
╙B╨
<__inference_signature_wrapper_function_with_signature_435846"П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в 
f
0
1
2
3
4
5
6
7
8
9"
trackable_list_wrapper
f
0
1
2
3
4
5
6
7
8
9"
trackable_list_wrapper
 "
trackable_list_wrapper
н
*non_trainable_variables

+layers
,metrics
-layer_regularization_losses
.layer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses"
_generic_user_object
╫2╘╤
╩▓╞
FullArgSpecD
args<Ъ9
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaultsв	
в 
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╫2╘╤
╩▓╞
FullArgSpecD
args<Ъ9
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaultsв	
в 
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
Є
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses
5_state_spec
6_input_encoder
7_lstm_network
8_output_encoder"
_tf_keras_layer
═
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses
?_means_projection_layer
	@_bias"
_tf_keras_layer
 "
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
 "
trackable_list_wrapper
н
Anon_trainable_variables

Blayers
Cmetrics
Dlayer_regularization_losses
Elayer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses"
_generic_user_object
╫2╘╤
╩▓╞
FullArgSpecD
args<Ъ9
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaultsв	
в 
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╫2╘╤
╩▓╞
FullArgSpecD
args<Ъ9
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaultsв	
в 
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
┴
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
J__call__
*K&call_and_return_all_conditional_losses
L_postprocessing_layers"
_tf_keras_layer
п
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q__call__
*R&call_and_return_all_conditional_losses
Scell"
_tf_keras_layer
'
T0"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
н
Unon_trainable_variables

Vlayers
Wmetrics
Xlayer_regularization_losses
Ylayer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses"
_generic_user_object
╔2╞├
╝▓╕
FullArgSpec7
args/Ъ,
jinputs
j
outer_rank

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╔2╞├
╝▓╕
FullArgSpec7
args/Ъ,
jinputs
j
outer_rank

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╗
Z	variables
[trainable_variables
\regularization_losses
]	keras_api
^__call__
*_&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
п
`	variables
atrainable_variables
bregularization_losses
c	keras_api
d__call__
*e&call_and_return_all_conditional_losses
bias"
_tf_keras_layer
 "
trackable_list_wrapper
5
60
71
T2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
н
fnon_trainable_variables

glayers
hmetrics
ilayer_regularization_losses
jlayer_metrics
F	variables
Gtrainable_variables
Hregularization_losses
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses"
_generic_user_object
█2╪╒
╬▓╩
FullArgSpecD
args<Ъ9
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaultsв

 
в 
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
█2╪╒
╬▓╩
FullArgSpecD
args<Ъ9
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaultsв

 
в 
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
.
k0
l1"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
н
mnon_trainable_variables

nlayers
ometrics
player_regularization_losses
qlayer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses"
_generic_user_object
╓2╙╨
╔▓┼
FullArgSpec@
args8Ъ5
jinputs
jinitial_state
j
reset_mask

jtraining
varargs
 
varkw
 
defaultsв

 

 
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╓2╙╨
╔▓┼
FullArgSpec@
args8Ъ5
jinputs
jinitial_state
j
reset_mask

jtraining
varargs
 
varkw
 
defaultsв

 

 
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
°
r	variables
strainable_variables
tregularization_losses
u	keras_api
v__call__
*w&call_and_return_all_conditional_losses
x_random_generator
y
state_size

kernel
recurrent_kernel
bias"
_tf_keras_layer
╗
z	variables
{trainable_variables
|regularization_losses
}	keras_api
~__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
 "
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Аnon_trainable_variables
Бlayers
Вmetrics
 Гlayer_regularization_losses
Дlayer_metrics
Z	variables
[trainable_variables
\regularization_losses
^__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses"
_generic_user_object
Ю2ЫШ
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ю2ЫШ
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Еnon_trainable_variables
Жlayers
Зmetrics
 Иlayer_regularization_losses
Йlayer_metrics
`	variables
atrainable_variables
bregularization_losses
d__call__
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses"
_generic_user_object
Ю2ЫШ
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ю2ЫШ
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
.
k0
l1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
л
К	variables
Лtrainable_variables
Мregularization_losses
Н	keras_api
О__call__
+П&call_and_return_all_conditional_losses"
_tf_keras_layer
┴
Р	variables
Сtrainable_variables
Тregularization_losses
У	keras_api
Ф__call__
+Х&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
 "
trackable_list_wrapper
'
S0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
5
0
1
2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Цnon_trainable_variables
Чlayers
Шmetrics
 Щlayer_regularization_losses
Ъlayer_metrics
r	variables
strainable_variables
tregularization_losses
v__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses"
_generic_user_object
╣2╢│
м▓и
FullArgSpec+
args#Ъ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╣2╢│
м▓и
FullArgSpec+
args#Ъ 
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
"
_generic_user_object
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Ыnon_trainable_variables
Ьlayers
Эmetrics
 Юlayer_regularization_losses
Яlayer_metrics
z	variables
{trainable_variables
|regularization_losses
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ю2ЫШ
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ю2ЫШ
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
╕
аnon_trainable_variables
бlayers
вmetrics
 гlayer_regularization_losses
дlayer_metrics
К	variables
Лtrainable_variables
Мregularization_losses
О__call__
+П&call_and_return_all_conditional_losses
'П"call_and_return_conditional_losses"
_generic_user_object
Ю2ЫШ
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ю2ЫШ
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
╕
еnon_trainable_variables
жlayers
зmetrics
 иlayer_regularization_losses
йlayer_metrics
Р	variables
Сtrainable_variables
Тregularization_losses
Ф__call__
+Х&call_and_return_all_conditional_losses
'Х"call_and_return_conditional_losses"
_generic_user_object
Ю2ЫШ
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ю2ЫШ
С▓Н
FullArgSpec
argsЪ

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
trackable_dict_wrapperC
__inference_<lambda>_435079$в

в 
к "К
unknown 	3
__inference_<lambda>_435081в

в 
к "к х
__inference_action_436075╟
ЬвШ
РвМ
╞▓┬
TimeStep,
	step_typeК
	step_type         &
rewardК
reward         *
discountК
discount         4
observation%К"
observation         
=Ъ:
К
0         g
К
1         g

 
к "Щ▓Х

PolicyStep&
actionК
action         R
stateIЪF
!К
state_0         g
!К
state_1         g
infoв з
__inference_action_436304Й
▐в┌
╥в╬
ю▓ъ
TimeStep6
	step_type)К&
time_step_step_type         0
reward&К#
time_step_reward         4
discount(К%
time_step_discount         >
observation/К,
time_step_observation         
WЪT
(К%
policy_state_0         g
(К%
policy_state_1         g

 
к "Щ▓Х

PolicyStep&
actionК
action         R
stateIЪF
!К
state_0         g
!К
state_1         g
infoв ┤
"__inference_distribution_fn_436493Н
ШвФ
МвИ
╞▓┬
TimeStep,
	step_typeК
	step_type         &
rewardК
reward         *
discountК
discount         4
observation%К"
observation         
=Ъ:
К
0         g
К
1         g
к "у▓▀

PolicyStepя
actionфТр│вп
`
?к<

locК         

scaleК         
Cк@

allow_nan_statsp

name
jNormal_1

validate_argsp 
в
j
parameters
в 
в
jname$tfp.distributions.Normal_ACTTypeSpec R
stateIЪF
!К
state_0         g
!К
state_1         g
infoв Щ
$__inference_get_initial_state_436509q"в
в
К

batch_size 
к "KЪH
"К
tensor_0         g
"К
tensor_1         gЧ
<__inference_signature_wrapper_function_with_signature_435797╓
╝в╕
в 
░км
2
arg_0_discount К

0/discount         
<
arg_0_observation'К$
0/observation         
.
arg_0_rewardК
0/reward         
4
arg_0_step_type!К
0/step_type         
(
arg_1_0К
1/0         g
(
arg_1_1К
1/1         g"ИкД
&
actionК
action         
,
state/0!К
state_0         g
,
state/1!К
state_1         g╩
<__inference_signature_wrapper_function_with_signature_435828Й0в-
в 
&к#
!

batch_sizeК

batch_size "UкR
'
0"К
tensor_0         g
'
1"К
tensor_1         gp
<__inference_signature_wrapper_function_with_signature_4358410в

в 
к "к

int64К
int64 	T
<__inference_signature_wrapper_function_with_signature_435846в

в 
к "к 