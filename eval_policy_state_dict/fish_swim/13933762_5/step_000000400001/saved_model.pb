Д╗
┐г
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
dtypetypeИ
╛
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
executor_typestring И
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.2.02unknown8╤Т
П
control_network/normalize/bVarHandleOp*
_output_shapes
: *
dtype0*
shape:м*,
shared_namecontrol_network/normalize/b
И
/control_network/normalize/b/Read/ReadVariableOpReadVariableOpcontrol_network/normalize/b*
_output_shapes	
:м*
dtype0
У
control_network/normalize/wVarHandleOp*
_output_shapes
: *
dtype0*
shape:	м*,
shared_namecontrol_network/normalize/w
М
/control_network/normalize/w/Read/ReadVariableOpReadVariableOpcontrol_network/normalize/w*
_output_shapes
:	м*
dtype0
Ы
!control_network/layer_norm/offsetVarHandleOp*
_output_shapes
: *
dtype0*
shape:м*2
shared_name#!control_network/layer_norm/offset
Ф
5control_network/layer_norm/offset/Read/ReadVariableOpReadVariableOp!control_network/layer_norm/offset*
_output_shapes	
:м*
dtype0
Щ
 control_network/layer_norm/scaleVarHandleOp*
_output_shapes
: *
dtype0*
shape:м*1
shared_name" control_network/layer_norm/scale
Т
4control_network/layer_norm/scale/Read/ReadVariableOpReadVariableOp control_network/layer_norm/scale*
_output_shapes	
:м*
dtype0
е
&LayerNormAndResidualMLP/mlp/linear_0/bVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*7
shared_name(&LayerNormAndResidualMLP/mlp/linear_0/b
Ю
:LayerNormAndResidualMLP/mlp/linear_0/b/Read/ReadVariableOpReadVariableOp&LayerNormAndResidualMLP/mlp/linear_0/b*
_output_shapes	
:А*
dtype0
к
&LayerNormAndResidualMLP/mlp/linear_0/wVarHandleOp*
_output_shapes
: *
dtype0*
shape:
мА*7
shared_name(&LayerNormAndResidualMLP/mlp/linear_0/w
г
:LayerNormAndResidualMLP/mlp/linear_0/w/Read/ReadVariableOpReadVariableOp&LayerNormAndResidualMLP/mlp/linear_0/w* 
_output_shapes
:
мА*
dtype0
й
(LayerNormAndResidualMLP/mlp/linear_0/b_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:А*9
shared_name*(LayerNormAndResidualMLP/mlp/linear_0/b_1
в
<LayerNormAndResidualMLP/mlp/linear_0/b_1/Read/ReadVariableOpReadVariableOp(LayerNormAndResidualMLP/mlp/linear_0/b_1*
_output_shapes	
:А*
dtype0
о
(LayerNormAndResidualMLP/mlp/linear_0/w_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*9
shared_name*(LayerNormAndResidualMLP/mlp/linear_0/w_1
з
<LayerNormAndResidualMLP/mlp/linear_0/w_1/Read/ReadVariableOpReadVariableOp(LayerNormAndResidualMLP/mlp/linear_0/w_1* 
_output_shapes
:
АА*
dtype0
е
&LayerNormAndResidualMLP/mlp/linear_1/bVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*7
shared_name(&LayerNormAndResidualMLP/mlp/linear_1/b
Ю
:LayerNormAndResidualMLP/mlp/linear_1/b/Read/ReadVariableOpReadVariableOp&LayerNormAndResidualMLP/mlp/linear_1/b*
_output_shapes	
:А*
dtype0
к
&LayerNormAndResidualMLP/mlp/linear_1/wVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*7
shared_name(&LayerNormAndResidualMLP/mlp/linear_1/w
г
:LayerNormAndResidualMLP/mlp/linear_1/w/Read/ReadVariableOpReadVariableOp&LayerNormAndResidualMLP/mlp/linear_1/w* 
_output_shapes
:
АА*
dtype0
▌
BLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/offsetVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*S
shared_nameDBLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/offset
╓
VLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/offset/Read/ReadVariableOpReadVariableOpBLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/offset*
_output_shapes	
:А*
dtype0
█
ALayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/scaleVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*R
shared_nameCALayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/scale
╘
ULayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/scale/Read/ReadVariableOpReadVariableOpALayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/scale*
_output_shapes	
:А*
dtype0
й
(LayerNormAndResidualMLP/mlp/linear_0/b_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:А*9
shared_name*(LayerNormAndResidualMLP/mlp/linear_0/b_2
в
<LayerNormAndResidualMLP/mlp/linear_0/b_2/Read/ReadVariableOpReadVariableOp(LayerNormAndResidualMLP/mlp/linear_0/b_2*
_output_shapes	
:А*
dtype0
о
(LayerNormAndResidualMLP/mlp/linear_0/w_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*9
shared_name*(LayerNormAndResidualMLP/mlp/linear_0/w_2
з
<LayerNormAndResidualMLP/mlp/linear_0/w_2/Read/ReadVariableOpReadVariableOp(LayerNormAndResidualMLP/mlp/linear_0/w_2* 
_output_shapes
:
АА*
dtype0
й
(LayerNormAndResidualMLP/mlp/linear_1/b_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:А*9
shared_name*(LayerNormAndResidualMLP/mlp/linear_1/b_1
в
<LayerNormAndResidualMLP/mlp/linear_1/b_1/Read/ReadVariableOpReadVariableOp(LayerNormAndResidualMLP/mlp/linear_1/b_1*
_output_shapes	
:А*
dtype0
о
(LayerNormAndResidualMLP/mlp/linear_1/w_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*9
shared_name*(LayerNormAndResidualMLP/mlp/linear_1/w_1
з
<LayerNormAndResidualMLP/mlp/linear_1/w_1/Read/ReadVariableOpReadVariableOp(LayerNormAndResidualMLP/mlp/linear_1/w_1* 
_output_shapes
:
АА*
dtype0
с
DLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/offset_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:А*U
shared_nameFDLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/offset_1
┌
XLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/offset_1/Read/ReadVariableOpReadVariableOpDLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/offset_1*
_output_shapes	
:А*
dtype0
▀
CLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/scale_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:А*T
shared_nameECLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/scale_1
╪
WLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/scale_1/Read/ReadVariableOpReadVariableOpCLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/scale_1*
_output_shapes	
:А*
dtype0
й
(LayerNormAndResidualMLP/mlp/linear_0/b_3VarHandleOp*
_output_shapes
: *
dtype0*
shape:А*9
shared_name*(LayerNormAndResidualMLP/mlp/linear_0/b_3
в
<LayerNormAndResidualMLP/mlp/linear_0/b_3/Read/ReadVariableOpReadVariableOp(LayerNormAndResidualMLP/mlp/linear_0/b_3*
_output_shapes	
:А*
dtype0
о
(LayerNormAndResidualMLP/mlp/linear_0/w_3VarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*9
shared_name*(LayerNormAndResidualMLP/mlp/linear_0/w_3
з
<LayerNormAndResidualMLP/mlp/linear_0/w_3/Read/ReadVariableOpReadVariableOp(LayerNormAndResidualMLP/mlp/linear_0/w_3* 
_output_shapes
:
АА*
dtype0
й
(LayerNormAndResidualMLP/mlp/linear_1/b_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:А*9
shared_name*(LayerNormAndResidualMLP/mlp/linear_1/b_2
в
<LayerNormAndResidualMLP/mlp/linear_1/b_2/Read/ReadVariableOpReadVariableOp(LayerNormAndResidualMLP/mlp/linear_1/b_2*
_output_shapes	
:А*
dtype0
о
(LayerNormAndResidualMLP/mlp/linear_1/w_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*9
shared_name*(LayerNormAndResidualMLP/mlp/linear_1/w_2
з
<LayerNormAndResidualMLP/mlp/linear_1/w_2/Read/ReadVariableOpReadVariableOp(LayerNormAndResidualMLP/mlp/linear_1/w_2* 
_output_shapes
:
АА*
dtype0
с
DLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/offset_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:А*U
shared_nameFDLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/offset_2
┌
XLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/offset_2/Read/ReadVariableOpReadVariableOpDLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/offset_2*
_output_shapes	
:А*
dtype0
▀
CLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/scale_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:А*T
shared_nameECLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/scale_2
╪
WLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/scale_2/Read/ReadVariableOpReadVariableOpCLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/scale_2*
_output_shapes	
:А*
dtype0
й
(LayerNormAndResidualMLP/mlp/linear_0/b_4VarHandleOp*
_output_shapes
: *
dtype0*
shape:А*9
shared_name*(LayerNormAndResidualMLP/mlp/linear_0/b_4
в
<LayerNormAndResidualMLP/mlp/linear_0/b_4/Read/ReadVariableOpReadVariableOp(LayerNormAndResidualMLP/mlp/linear_0/b_4*
_output_shapes	
:А*
dtype0
о
(LayerNormAndResidualMLP/mlp/linear_0/w_4VarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*9
shared_name*(LayerNormAndResidualMLP/mlp/linear_0/w_4
з
<LayerNormAndResidualMLP/mlp/linear_0/w_4/Read/ReadVariableOpReadVariableOp(LayerNormAndResidualMLP/mlp/linear_0/w_4* 
_output_shapes
:
АА*
dtype0
й
(LayerNormAndResidualMLP/mlp/linear_1/b_3VarHandleOp*
_output_shapes
: *
dtype0*
shape:А*9
shared_name*(LayerNormAndResidualMLP/mlp/linear_1/b_3
в
<LayerNormAndResidualMLP/mlp/linear_1/b_3/Read/ReadVariableOpReadVariableOp(LayerNormAndResidualMLP/mlp/linear_1/b_3*
_output_shapes	
:А*
dtype0
о
(LayerNormAndResidualMLP/mlp/linear_1/w_3VarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*9
shared_name*(LayerNormAndResidualMLP/mlp/linear_1/w_3
з
<LayerNormAndResidualMLP/mlp/linear_1/w_3/Read/ReadVariableOpReadVariableOp(LayerNormAndResidualMLP/mlp/linear_1/w_3* 
_output_shapes
:
АА*
dtype0
с
DLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/offset_3VarHandleOp*
_output_shapes
: *
dtype0*
shape:А*U
shared_nameFDLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/offset_3
┌
XLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/offset_3/Read/ReadVariableOpReadVariableOpDLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/offset_3*
_output_shapes	
:А*
dtype0
▀
CLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/scale_3VarHandleOp*
_output_shapes
: *
dtype0*
shape:А*T
shared_nameECLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/scale_3
╪
WLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/scale_3/Read/ReadVariableOpReadVariableOpCLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/scale_3*
_output_shapes	
:А*
dtype0
Ю
#MultivariateNormalDiagHead/linear/bVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#MultivariateNormalDiagHead/linear/b
Ч
7MultivariateNormalDiagHead/linear/b/Read/ReadVariableOpReadVariableOp#MultivariateNormalDiagHead/linear/b*
_output_shapes
:*
dtype0
г
#MultivariateNormalDiagHead/linear/wVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*4
shared_name%#MultivariateNormalDiagHead/linear/w
Ь
7MultivariateNormalDiagHead/linear/w/Read/ReadVariableOpReadVariableOp#MultivariateNormalDiagHead/linear/w*
_output_shapes
:	А*
dtype0
в
%MultivariateNormalDiagHead/linear/b_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%MultivariateNormalDiagHead/linear/b_1
Ы
9MultivariateNormalDiagHead/linear/b_1/Read/ReadVariableOpReadVariableOp%MultivariateNormalDiagHead/linear/b_1*
_output_shapes
:*
dtype0
з
%MultivariateNormalDiagHead/linear/w_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*6
shared_name'%MultivariateNormalDiagHead/linear/w_1
а
9MultivariateNormalDiagHead/linear/w_1/Read/ReadVariableOpReadVariableOp%MultivariateNormalDiagHead/linear/w_1*
_output_shapes
:	А*
dtype0

NoOpNoOp
Р"
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*╦!
value┴!B╛! B╖!
:

_variables
_trainable_variables

signatures
Ж
0
1
2
3
4
	5

6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
 28
!29
"30
#31
$32
%33
Ж
0
1
2
3
4
	5

6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
 28
!29
"30
#31
$32
%33
 
XV
VARIABLE_VALUEcontrol_network/normalize/b'_variables/0/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEcontrol_network/normalize/w'_variables/1/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!control_network/layer_norm/offset'_variables/2/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUE control_network/layer_norm/scale'_variables/3/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&LayerNormAndResidualMLP/mlp/linear_0/b'_variables/4/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&LayerNormAndResidualMLP/mlp/linear_0/w'_variables/5/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUE(LayerNormAndResidualMLP/mlp/linear_0/b_1'_variables/6/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUE(LayerNormAndResidualMLP/mlp/linear_0/w_1'_variables/7/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&LayerNormAndResidualMLP/mlp/linear_1/b'_variables/8/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE&LayerNormAndResidualMLP/mlp/linear_1/w'_variables/9/.ATTRIBUTES/VARIABLE_VALUE
А~
VARIABLE_VALUEBLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/offset(_variables/10/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEALayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/scale(_variables/11/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUE(LayerNormAndResidualMLP/mlp/linear_0/b_2(_variables/12/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUE(LayerNormAndResidualMLP/mlp/linear_0/w_2(_variables/13/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUE(LayerNormAndResidualMLP/mlp/linear_1/b_1(_variables/14/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUE(LayerNormAndResidualMLP/mlp/linear_1/w_1(_variables/15/.ATTRIBUTES/VARIABLE_VALUE
ГА
VARIABLE_VALUEDLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/offset_1(_variables/16/.ATTRIBUTES/VARIABLE_VALUE
Б
VARIABLE_VALUECLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/scale_1(_variables/17/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUE(LayerNormAndResidualMLP/mlp/linear_0/b_3(_variables/18/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUE(LayerNormAndResidualMLP/mlp/linear_0/w_3(_variables/19/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUE(LayerNormAndResidualMLP/mlp/linear_1/b_2(_variables/20/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUE(LayerNormAndResidualMLP/mlp/linear_1/w_2(_variables/21/.ATTRIBUTES/VARIABLE_VALUE
ГА
VARIABLE_VALUEDLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/offset_2(_variables/22/.ATTRIBUTES/VARIABLE_VALUE
Б
VARIABLE_VALUECLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/scale_2(_variables/23/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUE(LayerNormAndResidualMLP/mlp/linear_0/b_4(_variables/24/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUE(LayerNormAndResidualMLP/mlp/linear_0/w_4(_variables/25/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUE(LayerNormAndResidualMLP/mlp/linear_1/b_3(_variables/26/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUE(LayerNormAndResidualMLP/mlp/linear_1/w_3(_variables/27/.ATTRIBUTES/VARIABLE_VALUE
ГА
VARIABLE_VALUEDLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/offset_3(_variables/28/.ATTRIBUTES/VARIABLE_VALUE
Б
VARIABLE_VALUECLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/scale_3(_variables/29/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE#MultivariateNormalDiagHead/linear/b(_variables/30/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE#MultivariateNormalDiagHead/linear/w(_variables/31/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE%MultivariateNormalDiagHead/linear/b_1(_variables/32/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE%MultivariateNormalDiagHead/linear/w_1(_variables/33/.ATTRIBUTES/VARIABLE_VALUE
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
е
StatefulPartitionedCallStatefulPartitionedCallsaver_filename/control_network/normalize/b/Read/ReadVariableOp/control_network/normalize/w/Read/ReadVariableOp5control_network/layer_norm/offset/Read/ReadVariableOp4control_network/layer_norm/scale/Read/ReadVariableOp:LayerNormAndResidualMLP/mlp/linear_0/b/Read/ReadVariableOp:LayerNormAndResidualMLP/mlp/linear_0/w/Read/ReadVariableOp<LayerNormAndResidualMLP/mlp/linear_0/b_1/Read/ReadVariableOp<LayerNormAndResidualMLP/mlp/linear_0/w_1/Read/ReadVariableOp:LayerNormAndResidualMLP/mlp/linear_1/b/Read/ReadVariableOp:LayerNormAndResidualMLP/mlp/linear_1/w/Read/ReadVariableOpVLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/offset/Read/ReadVariableOpULayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/scale/Read/ReadVariableOp<LayerNormAndResidualMLP/mlp/linear_0/b_2/Read/ReadVariableOp<LayerNormAndResidualMLP/mlp/linear_0/w_2/Read/ReadVariableOp<LayerNormAndResidualMLP/mlp/linear_1/b_1/Read/ReadVariableOp<LayerNormAndResidualMLP/mlp/linear_1/w_1/Read/ReadVariableOpXLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/offset_1/Read/ReadVariableOpWLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/scale_1/Read/ReadVariableOp<LayerNormAndResidualMLP/mlp/linear_0/b_3/Read/ReadVariableOp<LayerNormAndResidualMLP/mlp/linear_0/w_3/Read/ReadVariableOp<LayerNormAndResidualMLP/mlp/linear_1/b_2/Read/ReadVariableOp<LayerNormAndResidualMLP/mlp/linear_1/w_2/Read/ReadVariableOpXLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/offset_2/Read/ReadVariableOpWLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/scale_2/Read/ReadVariableOp<LayerNormAndResidualMLP/mlp/linear_0/b_4/Read/ReadVariableOp<LayerNormAndResidualMLP/mlp/linear_0/w_4/Read/ReadVariableOp<LayerNormAndResidualMLP/mlp/linear_1/b_3/Read/ReadVariableOp<LayerNormAndResidualMLP/mlp/linear_1/w_3/Read/ReadVariableOpXLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/offset_3/Read/ReadVariableOpWLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/scale_3/Read/ReadVariableOp7MultivariateNormalDiagHead/linear/b/Read/ReadVariableOp7MultivariateNormalDiagHead/linear/w/Read/ReadVariableOp9MultivariateNormalDiagHead/linear/b_1/Read/ReadVariableOp9MultivariateNormalDiagHead/linear/w_1/Read/ReadVariableOpConst*/
Tin(
&2$*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *C
config_proto31

CPU

GPU 

TPU


TPU_SYSTEM2J 8*)
f$R"
 __inference__traced_save_2551863
·
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamecontrol_network/normalize/bcontrol_network/normalize/w!control_network/layer_norm/offset control_network/layer_norm/scale&LayerNormAndResidualMLP/mlp/linear_0/b&LayerNormAndResidualMLP/mlp/linear_0/w(LayerNormAndResidualMLP/mlp/linear_0/b_1(LayerNormAndResidualMLP/mlp/linear_0/w_1&LayerNormAndResidualMLP/mlp/linear_1/b&LayerNormAndResidualMLP/mlp/linear_1/wBLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/offsetALayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/scale(LayerNormAndResidualMLP/mlp/linear_0/b_2(LayerNormAndResidualMLP/mlp/linear_0/w_2(LayerNormAndResidualMLP/mlp/linear_1/b_1(LayerNormAndResidualMLP/mlp/linear_1/w_1DLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/offset_1CLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/scale_1(LayerNormAndResidualMLP/mlp/linear_0/b_3(LayerNormAndResidualMLP/mlp/linear_0/w_3(LayerNormAndResidualMLP/mlp/linear_1/b_2(LayerNormAndResidualMLP/mlp/linear_1/w_2DLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/offset_2CLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/scale_2(LayerNormAndResidualMLP/mlp/linear_0/b_4(LayerNormAndResidualMLP/mlp/linear_0/w_4(LayerNormAndResidualMLP/mlp/linear_1/b_3(LayerNormAndResidualMLP/mlp/linear_1/w_3DLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/offset_3CLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/scale_3#MultivariateNormalDiagHead/linear/b#MultivariateNormalDiagHead/linear/w%MultivariateNormalDiagHead/linear/b_1%MultivariateNormalDiagHead/linear/w_1*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *C
config_proto31

CPU

GPU 

TPU


TPU_SYSTEM2J 8*,
f'R%
#__inference__traced_restore_2551977гэ
щм
Г
__inference_wrapped_module_8836

args_0
args_0_1
args_0_2
args_0_3<
8control_network_normalize_matmul_readvariableop_resource9
5control_network_normalize_add_readvariableop_resourceD
@control_network_layer_norm_batchnorm_mul_readvariableop_resourceI
Econtrol_network_layer_norm_batchnorm_identity_readvariableop_resourceG
Clayernormandresidualmlp_mlp_linear_0_matmul_readvariableop_resourceD
@layernormandresidualmlp_mlp_linear_0_add_readvariableop_resourceI
Elayernormandresidualmlp_mlp_linear_0_matmul_1_readvariableop_resourceF
Blayernormandresidualmlp_mlp_linear_0_add_1_readvariableop_resourceG
Clayernormandresidualmlp_mlp_linear_1_matmul_readvariableop_resourceD
@layernormandresidualmlp_mlp_linear_1_add_readvariableop_resourcee
alayernormandresidualmlp_residuallayernormwrapper_layer_norm_batchnorm_mul_readvariableop_resourcej
flayernormandresidualmlp_residuallayernormwrapper_layer_norm_batchnorm_identity_readvariableop_resourceI
Elayernormandresidualmlp_mlp_linear_0_matmul_2_readvariableop_resourceF
Blayernormandresidualmlp_mlp_linear_0_add_2_readvariableop_resourceI
Elayernormandresidualmlp_mlp_linear_1_matmul_1_readvariableop_resourceF
Blayernormandresidualmlp_mlp_linear_1_add_1_readvariableop_resourceg
clayernormandresidualmlp_residuallayernormwrapper_layer_norm_batchnorm_1_mul_readvariableop_resourcel
hlayernormandresidualmlp_residuallayernormwrapper_layer_norm_batchnorm_1_identity_readvariableop_resourceI
Elayernormandresidualmlp_mlp_linear_0_matmul_3_readvariableop_resourceF
Blayernormandresidualmlp_mlp_linear_0_add_3_readvariableop_resourceI
Elayernormandresidualmlp_mlp_linear_1_matmul_2_readvariableop_resourceF
Blayernormandresidualmlp_mlp_linear_1_add_2_readvariableop_resourceg
clayernormandresidualmlp_residuallayernormwrapper_layer_norm_batchnorm_2_mul_readvariableop_resourcel
hlayernormandresidualmlp_residuallayernormwrapper_layer_norm_batchnorm_2_identity_readvariableop_resourceI
Elayernormandresidualmlp_mlp_linear_0_matmul_4_readvariableop_resourceF
Blayernormandresidualmlp_mlp_linear_0_add_4_readvariableop_resourceI
Elayernormandresidualmlp_mlp_linear_1_matmul_3_readvariableop_resourceF
Blayernormandresidualmlp_mlp_linear_1_add_3_readvariableop_resourceg
clayernormandresidualmlp_residuallayernormwrapper_layer_norm_batchnorm_3_mul_readvariableop_resourcel
hlayernormandresidualmlp_residuallayernormwrapper_layer_norm_batchnorm_3_identity_readvariableop_resourceD
@multivariatenormaldiaghead_linear_matmul_readvariableop_resourceA
=multivariatenormaldiaghead_linear_add_readvariableop_resourceF
Bmultivariatenormaldiaghead_linear_matmul_1_readvariableop_resourceC
?multivariatenormaldiaghead_linear_add_1_readvariableop_resource
identityИt
control_network/flatten/ShapeShapeargs_0*
T0*
_output_shapes
:2
control_network/flatten/Shapeд
+control_network/flatten/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+control_network/flatten/strided_slice/stackи
-control_network/flatten/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-control_network/flatten/strided_slice/stack_1и
-control_network/flatten/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-control_network/flatten/strided_slice/stack_2Ё
%control_network/flatten/strided_sliceStridedSlice&control_network/flatten/Shape:output:04control_network/flatten/strided_slice/stack:output:06control_network/flatten/strided_slice/stack_1:output:06control_network/flatten/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2'
%control_network/flatten/strided_sliceЬ
'control_network/flatten/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'control_network/flatten/concat/values_1М
#control_network/flatten/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#control_network/flatten/concat/axisК
control_network/flatten/concatConcatV2.control_network/flatten/strided_slice:output:00control_network/flatten/concat/values_1:output:0,control_network/flatten/concat/axis:output:0*
N*
T0*
_output_shapes
:2 
control_network/flatten/concat░
control_network/flatten/ReshapeReshapeargs_0'control_network/flatten/concat:output:0*
T0*'
_output_shapes
:         2!
control_network/flatten/Reshapez
control_network/flatten_1/ShapeShapeargs_0_1*
T0*
_output_shapes
:2!
control_network/flatten_1/Shapeи
-control_network/flatten_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-control_network/flatten_1/strided_slice/stackм
/control_network/flatten_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/control_network/flatten_1/strided_slice/stack_1м
/control_network/flatten_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/control_network/flatten_1/strided_slice/stack_2№
'control_network/flatten_1/strided_sliceStridedSlice(control_network/flatten_1/Shape:output:06control_network/flatten_1/strided_slice/stack:output:08control_network/flatten_1/strided_slice/stack_1:output:08control_network/flatten_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2)
'control_network/flatten_1/strided_sliceа
)control_network/flatten_1/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)control_network/flatten_1/concat/values_1Р
%control_network/flatten_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2'
%control_network/flatten_1/concat/axisФ
 control_network/flatten_1/concatConcatV20control_network/flatten_1/strided_slice:output:02control_network/flatten_1/concat/values_1:output:0.control_network/flatten_1/concat/axis:output:0*
N*
T0*
_output_shapes
:2"
 control_network/flatten_1/concat╕
!control_network/flatten_1/ReshapeReshapeargs_0_1)control_network/flatten_1/concat:output:0*
T0*'
_output_shapes
:         2#
!control_network/flatten_1/Reshapez
control_network/flatten_2/ShapeShapeargs_0_2*
T0*
_output_shapes
:2!
control_network/flatten_2/Shapeи
-control_network/flatten_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-control_network/flatten_2/strided_slice/stackм
/control_network/flatten_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/control_network/flatten_2/strided_slice/stack_1м
/control_network/flatten_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/control_network/flatten_2/strided_slice/stack_2№
'control_network/flatten_2/strided_sliceStridedSlice(control_network/flatten_2/Shape:output:06control_network/flatten_2/strided_slice/stack:output:08control_network/flatten_2/strided_slice/stack_1:output:08control_network/flatten_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2)
'control_network/flatten_2/strided_sliceа
)control_network/flatten_2/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)control_network/flatten_2/concat/values_1Р
%control_network/flatten_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2'
%control_network/flatten_2/concat/axisФ
 control_network/flatten_2/concatConcatV20control_network/flatten_2/strided_slice:output:02control_network/flatten_2/concat/values_1:output:0.control_network/flatten_2/concat/axis:output:0*
N*
T0*
_output_shapes
:2"
 control_network/flatten_2/concat╕
!control_network/flatten_2/ReshapeReshapeargs_0_2)control_network/flatten_2/concat:output:0*
T0*'
_output_shapes
:         2#
!control_network/flatten_2/Reshapez
control_network/flatten_3/ShapeShapeargs_0_3*
T0*
_output_shapes
:2!
control_network/flatten_3/Shapeи
-control_network/flatten_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-control_network/flatten_3/strided_slice/stackм
/control_network/flatten_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/control_network/flatten_3/strided_slice/stack_1м
/control_network/flatten_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/control_network/flatten_3/strided_slice/stack_2№
'control_network/flatten_3/strided_sliceStridedSlice(control_network/flatten_3/Shape:output:06control_network/flatten_3/strided_slice/stack:output:08control_network/flatten_3/strided_slice/stack_1:output:08control_network/flatten_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2)
'control_network/flatten_3/strided_sliceа
)control_network/flatten_3/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)control_network/flatten_3/concat/values_1Р
%control_network/flatten_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2'
%control_network/flatten_3/concat/axisФ
 control_network/flatten_3/concatConcatV20control_network/flatten_3/strided_slice:output:02control_network/flatten_3/concat/values_1:output:0.control_network/flatten_3/concat/axis:output:0*
N*
T0*
_output_shapes
:2"
 control_network/flatten_3/concat╕
!control_network/flatten_3/ReshapeReshapeargs_0_3)control_network/flatten_3/concat:output:0*
T0*'
_output_shapes
:         2#
!control_network/flatten_3/Reshape|
control_network/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
control_network/concat/axis╦
control_network/concatConcatV2(control_network/flatten/Reshape:output:0*control_network/flatten_1/Reshape:output:0*control_network/flatten_2/Reshape:output:0*control_network/flatten_3/Reshape:output:0$control_network/concat/axis:output:0*
N*
T0*'
_output_shapes
:         2
control_network/concat▄
/control_network/normalize/MatMul/ReadVariableOpReadVariableOp8control_network_normalize_matmul_readvariableop_resource*
_output_shapes
:	м*
dtype021
/control_network/normalize/MatMul/ReadVariableOp█
 control_network/normalize/MatMulMatMulcontrol_network/concat:output:07control_network/normalize/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         м2"
 control_network/normalize/MatMul╧
,control_network/normalize/Add/ReadVariableOpReadVariableOp5control_network_normalize_add_readvariableop_resource*
_output_shapes	
:м*
dtype02.
,control_network/normalize/Add/ReadVariableOp┌
control_network/normalize/AddAdd*control_network/normalize/MatMul:product:04control_network/normalize/Add/ReadVariableOp:value:0*
T0*(
_output_shapes
:         м2
control_network/normalize/Add└
9control_network/layer_norm/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9control_network/layer_norm/moments/mean/reduction_indicesД
'control_network/layer_norm/moments/meanMean!control_network/normalize/Add:z:0Bcontrol_network/layer_norm/moments/mean/reduction_indices:output:0*
T0*'
_output_shapes
:         *
	keep_dims(2)
'control_network/layer_norm/moments/mean╓
/control_network/layer_norm/moments/StopGradientStopGradient0control_network/layer_norm/moments/mean:output:0*
T0*'
_output_shapes
:         21
/control_network/layer_norm/moments/StopGradientС
4control_network/layer_norm/moments/SquaredDifferenceSquaredDifference!control_network/normalize/Add:z:08control_network/layer_norm/moments/StopGradient:output:0*
T0*(
_output_shapes
:         м26
4control_network/layer_norm/moments/SquaredDifference╚
=control_network/layer_norm/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2?
=control_network/layer_norm/moments/variance/reduction_indicesз
+control_network/layer_norm/moments/varianceMean8control_network/layer_norm/moments/SquaredDifference:z:0Fcontrol_network/layer_norm/moments/variance/reduction_indices:output:0*
T0*'
_output_shapes
:         *
	keep_dims(2-
+control_network/layer_norm/moments/varianceЭ
*control_network/layer_norm/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *м┼'72,
*control_network/layer_norm/batchnorm/add/y·
(control_network/layer_norm/batchnorm/addAddV24control_network/layer_norm/moments/variance:output:03control_network/layer_norm/batchnorm/add/y:output:0*
T0*'
_output_shapes
:         2*
(control_network/layer_norm/batchnorm/add┴
*control_network/layer_norm/batchnorm/RsqrtRsqrt,control_network/layer_norm/batchnorm/add:z:0*
T0*'
_output_shapes
:         2,
*control_network/layer_norm/batchnorm/RsqrtЁ
7control_network/layer_norm/batchnorm/mul/ReadVariableOpReadVariableOp@control_network_layer_norm_batchnorm_mul_readvariableop_resource*
_output_shapes	
:м*
dtype029
7control_network/layer_norm/batchnorm/mul/ReadVariableOp 
(control_network/layer_norm/batchnorm/mulMul.control_network/layer_norm/batchnorm/Rsqrt:y:0?control_network/layer_norm/batchnorm/mul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         м2*
(control_network/layer_norm/batchnorm/mulу
*control_network/layer_norm/batchnorm/mul_1Mul!control_network/normalize/Add:z:0,control_network/layer_norm/batchnorm/mul:z:0*
T0*(
_output_shapes
:         м2,
*control_network/layer_norm/batchnorm/mul_1Є
*control_network/layer_norm/batchnorm/mul_2Mul0control_network/layer_norm/moments/mean:output:0,control_network/layer_norm/batchnorm/mul:z:0*
T0*(
_output_shapes
:         м2,
*control_network/layer_norm/batchnorm/mul_2 
<control_network/layer_norm/batchnorm/Identity/ReadVariableOpReadVariableOpEcontrol_network_layer_norm_batchnorm_identity_readvariableop_resource*
_output_shapes	
:м*
dtype02>
<control_network/layer_norm/batchnorm/Identity/ReadVariableOp╓
-control_network/layer_norm/batchnorm/IdentityIdentityDcontrol_network/layer_norm/batchnorm/Identity/ReadVariableOp:value:0*
T0*
_output_shapes	
:м2/
-control_network/layer_norm/batchnorm/IdentityЎ
(control_network/layer_norm/batchnorm/subSub6control_network/layer_norm/batchnorm/Identity:output:0.control_network/layer_norm/batchnorm/mul_2:z:0*
T0*(
_output_shapes
:         м2*
(control_network/layer_norm/batchnorm/subЄ
*control_network/layer_norm/batchnorm/add_1AddV2.control_network/layer_norm/batchnorm/mul_1:z:0,control_network/layer_norm/batchnorm/sub:z:0*
T0*(
_output_shapes
:         м2,
*control_network/layer_norm/batchnorm/add_1Ч
control_network/TanhTanh.control_network/layer_norm/batchnorm/add_1:z:0*
T0*(
_output_shapes
:         м2
control_network/TanhМ
#control_network/concat_1/concat_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#control_network/concat_1/concat_dimЫ
control_network/concat_1/concatIdentitycontrol_network/Tanh:y:0*
T0*(
_output_shapes
:         м2!
control_network/concat_1/concat■
:LayerNormAndResidualMLP/mlp/linear_0/MatMul/ReadVariableOpReadVariableOpClayernormandresidualmlp_mlp_linear_0_matmul_readvariableop_resource* 
_output_shapes
:
мА*
dtype02<
:LayerNormAndResidualMLP/mlp/linear_0/MatMul/ReadVariableOpЕ
+LayerNormAndResidualMLP/mlp/linear_0/MatMulMatMul(control_network/concat_1/concat:output:0BLayerNormAndResidualMLP/mlp/linear_0/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2-
+LayerNormAndResidualMLP/mlp/linear_0/MatMulЁ
7LayerNormAndResidualMLP/mlp/linear_0/Add/ReadVariableOpReadVariableOp@layernormandresidualmlp_mlp_linear_0_add_readvariableop_resource*
_output_shapes	
:А*
dtype029
7LayerNormAndResidualMLP/mlp/linear_0/Add/ReadVariableOpЖ
(LayerNormAndResidualMLP/mlp/linear_0/AddAdd5LayerNormAndResidualMLP/mlp/linear_0/MatMul:product:0?LayerNormAndResidualMLP/mlp/linear_0/Add/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2*
(LayerNormAndResidualMLP/mlp/linear_0/AddД
<LayerNormAndResidualMLP/mlp/linear_0/MatMul_1/ReadVariableOpReadVariableOpElayernormandresidualmlp_mlp_linear_0_matmul_1_readvariableop_resource* 
_output_shapes
:
АА*
dtype02>
<LayerNormAndResidualMLP/mlp/linear_0/MatMul_1/ReadVariableOpП
-LayerNormAndResidualMLP/mlp/linear_0/MatMul_1MatMul,LayerNormAndResidualMLP/mlp/linear_0/Add:z:0DLayerNormAndResidualMLP/mlp/linear_0/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2/
-LayerNormAndResidualMLP/mlp/linear_0/MatMul_1Ў
9LayerNormAndResidualMLP/mlp/linear_0/Add_1/ReadVariableOpReadVariableOpBlayernormandresidualmlp_mlp_linear_0_add_1_readvariableop_resource*
_output_shapes	
:А*
dtype02;
9LayerNormAndResidualMLP/mlp/linear_0/Add_1/ReadVariableOpО
*LayerNormAndResidualMLP/mlp/linear_0/Add_1Add7LayerNormAndResidualMLP/mlp/linear_0/MatMul_1:product:0ALayerNormAndResidualMLP/mlp/linear_0/Add_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2,
*LayerNormAndResidualMLP/mlp/linear_0/Add_1п
 LayerNormAndResidualMLP/mlp/ReluRelu.LayerNormAndResidualMLP/mlp/linear_0/Add_1:z:0*
T0*(
_output_shapes
:         А2"
 LayerNormAndResidualMLP/mlp/Relu■
:LayerNormAndResidualMLP/mlp/linear_1/MatMul/ReadVariableOpReadVariableOpClayernormandresidualmlp_mlp_linear_1_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02<
:LayerNormAndResidualMLP/mlp/linear_1/MatMul/ReadVariableOpЛ
+LayerNormAndResidualMLP/mlp/linear_1/MatMulMatMul.LayerNormAndResidualMLP/mlp/Relu:activations:0BLayerNormAndResidualMLP/mlp/linear_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2-
+LayerNormAndResidualMLP/mlp/linear_1/MatMulЁ
7LayerNormAndResidualMLP/mlp/linear_1/Add/ReadVariableOpReadVariableOp@layernormandresidualmlp_mlp_linear_1_add_readvariableop_resource*
_output_shapes	
:А*
dtype029
7LayerNormAndResidualMLP/mlp/linear_1/Add/ReadVariableOpЖ
(LayerNormAndResidualMLP/mlp/linear_1/AddAdd5LayerNormAndResidualMLP/mlp/linear_1/MatMul:product:0?LayerNormAndResidualMLP/mlp/linear_1/Add/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2*
(LayerNormAndResidualMLP/mlp/linear_1/AddД
4LayerNormAndResidualMLP/ResidualLayernormWrapper/addAddV2,LayerNormAndResidualMLP/mlp/linear_1/Add:z:0,LayerNormAndResidualMLP/mlp/linear_0/Add:z:0*
T0*(
_output_shapes
:         А26
4LayerNormAndResidualMLP/ResidualLayernormWrapper/addЛ
ZLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
         2\
ZLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments/mean/reduction_indices■
HLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments/meanMean8LayerNormAndResidualMLP/ResidualLayernormWrapper/add:z:0cLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments/mean/reduction_indices:output:0*
T0*'
_output_shapes
:         *
	keep_dims(2J
HLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments/mean╣
PLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments/StopGradientStopGradientQLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments/mean:output:0*
T0*'
_output_shapes
:         2R
PLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments/StopGradientЛ
ULayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments/SquaredDifferenceSquaredDifference8LayerNormAndResidualMLP/ResidualLayernormWrapper/add:z:0YLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments/StopGradient:output:0*
T0*(
_output_shapes
:         А2W
ULayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments/SquaredDifferenceУ
^LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
         2`
^LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments/variance/reduction_indicesл
LLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments/varianceMeanYLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments/SquaredDifference:z:0gLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments/variance/reduction_indices:output:0*
T0*'
_output_shapes
:         *
	keep_dims(2N
LLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments/variance▀
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *м┼'72M
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/add/y■
ILayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/addAddV2ULayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments/variance:output:0TLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/add/y:output:0*
T0*'
_output_shapes
:         2K
ILayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/addд
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/RsqrtRsqrtMLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/add:z:0*
T0*'
_output_shapes
:         2M
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/Rsqrt╙
XLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/mul/ReadVariableOpReadVariableOpalayernormandresidualmlp_residuallayernormwrapper_layer_norm_batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02Z
XLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/mul/ReadVariableOpГ
ILayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/mulMulOLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/Rsqrt:y:0`LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/mul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2K
ILayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/mul▌
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/mul_1Mul8LayerNormAndResidualMLP/ResidualLayernormWrapper/add:z:0MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/mul:z:0*
T0*(
_output_shapes
:         А2M
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/mul_1Ў
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/mul_2MulQLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments/mean:output:0MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/mul:z:0*
T0*(
_output_shapes
:         А2M
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/mul_2т
]LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/Identity/ReadVariableOpReadVariableOpflayernormandresidualmlp_residuallayernormwrapper_layer_norm_batchnorm_identity_readvariableop_resource*
_output_shapes	
:А*
dtype02_
]LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/Identity/ReadVariableOp╣
NLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/IdentityIdentityeLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/Identity/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2P
NLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/Identity·
ILayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/subSubWLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/Identity:output:0OLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/mul_2:z:0*
T0*(
_output_shapes
:         А2K
ILayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/subЎ
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/add_1AddV2OLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/mul_1:z:0MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/sub:z:0*
T0*(
_output_shapes
:         А2M
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/add_1Д
<LayerNormAndResidualMLP/mlp/linear_0/MatMul_2/ReadVariableOpReadVariableOpElayernormandresidualmlp_mlp_linear_0_matmul_2_readvariableop_resource* 
_output_shapes
:
АА*
dtype02>
<LayerNormAndResidualMLP/mlp/linear_0/MatMul_2/ReadVariableOp▓
-LayerNormAndResidualMLP/mlp/linear_0/MatMul_2MatMulOLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/add_1:z:0DLayerNormAndResidualMLP/mlp/linear_0/MatMul_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2/
-LayerNormAndResidualMLP/mlp/linear_0/MatMul_2Ў
9LayerNormAndResidualMLP/mlp/linear_0/Add_2/ReadVariableOpReadVariableOpBlayernormandresidualmlp_mlp_linear_0_add_2_readvariableop_resource*
_output_shapes	
:А*
dtype02;
9LayerNormAndResidualMLP/mlp/linear_0/Add_2/ReadVariableOpО
*LayerNormAndResidualMLP/mlp/linear_0/Add_2Add7LayerNormAndResidualMLP/mlp/linear_0/MatMul_2:product:0ALayerNormAndResidualMLP/mlp/linear_0/Add_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2,
*LayerNormAndResidualMLP/mlp/linear_0/Add_2│
"LayerNormAndResidualMLP/mlp/Relu_1Relu.LayerNormAndResidualMLP/mlp/linear_0/Add_2:z:0*
T0*(
_output_shapes
:         А2$
"LayerNormAndResidualMLP/mlp/Relu_1Д
<LayerNormAndResidualMLP/mlp/linear_1/MatMul_1/ReadVariableOpReadVariableOpElayernormandresidualmlp_mlp_linear_1_matmul_1_readvariableop_resource* 
_output_shapes
:
АА*
dtype02>
<LayerNormAndResidualMLP/mlp/linear_1/MatMul_1/ReadVariableOpУ
-LayerNormAndResidualMLP/mlp/linear_1/MatMul_1MatMul0LayerNormAndResidualMLP/mlp/Relu_1:activations:0DLayerNormAndResidualMLP/mlp/linear_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2/
-LayerNormAndResidualMLP/mlp/linear_1/MatMul_1Ў
9LayerNormAndResidualMLP/mlp/linear_1/Add_1/ReadVariableOpReadVariableOpBlayernormandresidualmlp_mlp_linear_1_add_1_readvariableop_resource*
_output_shapes	
:А*
dtype02;
9LayerNormAndResidualMLP/mlp/linear_1/Add_1/ReadVariableOpО
*LayerNormAndResidualMLP/mlp/linear_1/Add_1Add7LayerNormAndResidualMLP/mlp/linear_1/MatMul_1:product:0ALayerNormAndResidualMLP/mlp/linear_1/Add_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2,
*LayerNormAndResidualMLP/mlp/linear_1/Add_1н
6LayerNormAndResidualMLP/ResidualLayernormWrapper/add_1AddV2.LayerNormAndResidualMLP/mlp/linear_1/Add_1:z:0OLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/add_1:z:0*
T0*(
_output_shapes
:         А28
6LayerNormAndResidualMLP/ResidualLayernormWrapper/add_1П
\LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
         2^
\LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_1/mean/reduction_indicesЖ
JLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_1/meanMean:LayerNormAndResidualMLP/ResidualLayernormWrapper/add_1:z:0eLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_1/mean/reduction_indices:output:0*
T0*'
_output_shapes
:         *
	keep_dims(2L
JLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_1/mean┐
RLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_1/StopGradientStopGradientSLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_1/mean:output:0*
T0*'
_output_shapes
:         2T
RLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_1/StopGradientУ
WLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_1/SquaredDifferenceSquaredDifference:LayerNormAndResidualMLP/ResidualLayernormWrapper/add_1:z:0[LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_1/StopGradient:output:0*
T0*(
_output_shapes
:         А2Y
WLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_1/SquaredDifferenceЧ
`LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
         2b
`LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_1/variance/reduction_indices│
NLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_1/varianceMean[LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_1/SquaredDifference:z:0iLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_1/variance/reduction_indices:output:0*
T0*'
_output_shapes
:         *
	keep_dims(2P
NLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_1/varianceу
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *м┼'72O
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/add/yЖ
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/addAddV2WLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_1/variance:output:0VLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/add/y:output:0*
T0*'
_output_shapes
:         2M
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/addк
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/RsqrtRsqrtOLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/add:z:0*
T0*'
_output_shapes
:         2O
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/Rsqrt┘
ZLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/mul/ReadVariableOpReadVariableOpclayernormandresidualmlp_residuallayernormwrapper_layer_norm_batchnorm_1_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02\
ZLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/mul/ReadVariableOpЛ
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/mulMulQLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/Rsqrt:y:0bLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/mul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2M
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/mulх
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/mul_1Mul:LayerNormAndResidualMLP/ResidualLayernormWrapper/add_1:z:0OLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/mul:z:0*
T0*(
_output_shapes
:         А2O
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/mul_1■
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/mul_2MulSLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_1/mean:output:0OLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/mul:z:0*
T0*(
_output_shapes
:         А2O
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/mul_2ш
_LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/Identity/ReadVariableOpReadVariableOphlayernormandresidualmlp_residuallayernormwrapper_layer_norm_batchnorm_1_identity_readvariableop_resource*
_output_shapes	
:А*
dtype02a
_LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/Identity/ReadVariableOp┐
PLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/IdentityIdentitygLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/Identity/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2R
PLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/IdentityВ
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/subSubYLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/Identity:output:0QLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/mul_2:z:0*
T0*(
_output_shapes
:         А2M
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/sub■
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/add_1AddV2QLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/mul_1:z:0OLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/sub:z:0*
T0*(
_output_shapes
:         А2O
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/add_1Д
<LayerNormAndResidualMLP/mlp/linear_0/MatMul_3/ReadVariableOpReadVariableOpElayernormandresidualmlp_mlp_linear_0_matmul_3_readvariableop_resource* 
_output_shapes
:
АА*
dtype02>
<LayerNormAndResidualMLP/mlp/linear_0/MatMul_3/ReadVariableOp┤
-LayerNormAndResidualMLP/mlp/linear_0/MatMul_3MatMulQLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/add_1:z:0DLayerNormAndResidualMLP/mlp/linear_0/MatMul_3/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2/
-LayerNormAndResidualMLP/mlp/linear_0/MatMul_3Ў
9LayerNormAndResidualMLP/mlp/linear_0/Add_3/ReadVariableOpReadVariableOpBlayernormandresidualmlp_mlp_linear_0_add_3_readvariableop_resource*
_output_shapes	
:А*
dtype02;
9LayerNormAndResidualMLP/mlp/linear_0/Add_3/ReadVariableOpО
*LayerNormAndResidualMLP/mlp/linear_0/Add_3Add7LayerNormAndResidualMLP/mlp/linear_0/MatMul_3:product:0ALayerNormAndResidualMLP/mlp/linear_0/Add_3/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2,
*LayerNormAndResidualMLP/mlp/linear_0/Add_3│
"LayerNormAndResidualMLP/mlp/Relu_2Relu.LayerNormAndResidualMLP/mlp/linear_0/Add_3:z:0*
T0*(
_output_shapes
:         А2$
"LayerNormAndResidualMLP/mlp/Relu_2Д
<LayerNormAndResidualMLP/mlp/linear_1/MatMul_2/ReadVariableOpReadVariableOpElayernormandresidualmlp_mlp_linear_1_matmul_2_readvariableop_resource* 
_output_shapes
:
АА*
dtype02>
<LayerNormAndResidualMLP/mlp/linear_1/MatMul_2/ReadVariableOpУ
-LayerNormAndResidualMLP/mlp/linear_1/MatMul_2MatMul0LayerNormAndResidualMLP/mlp/Relu_2:activations:0DLayerNormAndResidualMLP/mlp/linear_1/MatMul_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2/
-LayerNormAndResidualMLP/mlp/linear_1/MatMul_2Ў
9LayerNormAndResidualMLP/mlp/linear_1/Add_2/ReadVariableOpReadVariableOpBlayernormandresidualmlp_mlp_linear_1_add_2_readvariableop_resource*
_output_shapes	
:А*
dtype02;
9LayerNormAndResidualMLP/mlp/linear_1/Add_2/ReadVariableOpО
*LayerNormAndResidualMLP/mlp/linear_1/Add_2Add7LayerNormAndResidualMLP/mlp/linear_1/MatMul_2:product:0ALayerNormAndResidualMLP/mlp/linear_1/Add_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2,
*LayerNormAndResidualMLP/mlp/linear_1/Add_2п
6LayerNormAndResidualMLP/ResidualLayernormWrapper/add_2AddV2.LayerNormAndResidualMLP/mlp/linear_1/Add_2:z:0QLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/add_1:z:0*
T0*(
_output_shapes
:         А28
6LayerNormAndResidualMLP/ResidualLayernormWrapper/add_2П
\LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_2/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
         2^
\LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_2/mean/reduction_indicesЖ
JLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_2/meanMean:LayerNormAndResidualMLP/ResidualLayernormWrapper/add_2:z:0eLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_2/mean/reduction_indices:output:0*
T0*'
_output_shapes
:         *
	keep_dims(2L
JLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_2/mean┐
RLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_2/StopGradientStopGradientSLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_2/mean:output:0*
T0*'
_output_shapes
:         2T
RLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_2/StopGradientУ
WLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_2/SquaredDifferenceSquaredDifference:LayerNormAndResidualMLP/ResidualLayernormWrapper/add_2:z:0[LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_2/StopGradient:output:0*
T0*(
_output_shapes
:         А2Y
WLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_2/SquaredDifferenceЧ
`LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_2/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
         2b
`LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_2/variance/reduction_indices│
NLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_2/varianceMean[LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_2/SquaredDifference:z:0iLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_2/variance/reduction_indices:output:0*
T0*'
_output_shapes
:         *
	keep_dims(2P
NLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_2/varianceу
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *м┼'72O
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/add/yЖ
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/addAddV2WLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_2/variance:output:0VLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/add/y:output:0*
T0*'
_output_shapes
:         2M
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/addк
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/RsqrtRsqrtOLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/add:z:0*
T0*'
_output_shapes
:         2O
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/Rsqrt┘
ZLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/mul/ReadVariableOpReadVariableOpclayernormandresidualmlp_residuallayernormwrapper_layer_norm_batchnorm_2_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02\
ZLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/mul/ReadVariableOpЛ
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/mulMulQLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/Rsqrt:y:0bLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/mul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2M
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/mulх
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/mul_1Mul:LayerNormAndResidualMLP/ResidualLayernormWrapper/add_2:z:0OLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/mul:z:0*
T0*(
_output_shapes
:         А2O
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/mul_1■
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/mul_2MulSLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_2/mean:output:0OLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/mul:z:0*
T0*(
_output_shapes
:         А2O
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/mul_2ш
_LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/Identity/ReadVariableOpReadVariableOphlayernormandresidualmlp_residuallayernormwrapper_layer_norm_batchnorm_2_identity_readvariableop_resource*
_output_shapes	
:А*
dtype02a
_LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/Identity/ReadVariableOp┐
PLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/IdentityIdentitygLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/Identity/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2R
PLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/IdentityВ
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/subSubYLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/Identity:output:0QLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/mul_2:z:0*
T0*(
_output_shapes
:         А2M
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/sub■
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/add_1AddV2QLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/mul_1:z:0OLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/sub:z:0*
T0*(
_output_shapes
:         А2O
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/add_1Д
<LayerNormAndResidualMLP/mlp/linear_0/MatMul_4/ReadVariableOpReadVariableOpElayernormandresidualmlp_mlp_linear_0_matmul_4_readvariableop_resource* 
_output_shapes
:
АА*
dtype02>
<LayerNormAndResidualMLP/mlp/linear_0/MatMul_4/ReadVariableOp┤
-LayerNormAndResidualMLP/mlp/linear_0/MatMul_4MatMulQLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/add_1:z:0DLayerNormAndResidualMLP/mlp/linear_0/MatMul_4/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2/
-LayerNormAndResidualMLP/mlp/linear_0/MatMul_4Ў
9LayerNormAndResidualMLP/mlp/linear_0/Add_4/ReadVariableOpReadVariableOpBlayernormandresidualmlp_mlp_linear_0_add_4_readvariableop_resource*
_output_shapes	
:А*
dtype02;
9LayerNormAndResidualMLP/mlp/linear_0/Add_4/ReadVariableOpО
*LayerNormAndResidualMLP/mlp/linear_0/Add_4Add7LayerNormAndResidualMLP/mlp/linear_0/MatMul_4:product:0ALayerNormAndResidualMLP/mlp/linear_0/Add_4/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2,
*LayerNormAndResidualMLP/mlp/linear_0/Add_4│
"LayerNormAndResidualMLP/mlp/Relu_3Relu.LayerNormAndResidualMLP/mlp/linear_0/Add_4:z:0*
T0*(
_output_shapes
:         А2$
"LayerNormAndResidualMLP/mlp/Relu_3Д
<LayerNormAndResidualMLP/mlp/linear_1/MatMul_3/ReadVariableOpReadVariableOpElayernormandresidualmlp_mlp_linear_1_matmul_3_readvariableop_resource* 
_output_shapes
:
АА*
dtype02>
<LayerNormAndResidualMLP/mlp/linear_1/MatMul_3/ReadVariableOpУ
-LayerNormAndResidualMLP/mlp/linear_1/MatMul_3MatMul0LayerNormAndResidualMLP/mlp/Relu_3:activations:0DLayerNormAndResidualMLP/mlp/linear_1/MatMul_3/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2/
-LayerNormAndResidualMLP/mlp/linear_1/MatMul_3Ў
9LayerNormAndResidualMLP/mlp/linear_1/Add_3/ReadVariableOpReadVariableOpBlayernormandresidualmlp_mlp_linear_1_add_3_readvariableop_resource*
_output_shapes	
:А*
dtype02;
9LayerNormAndResidualMLP/mlp/linear_1/Add_3/ReadVariableOpО
*LayerNormAndResidualMLP/mlp/linear_1/Add_3Add7LayerNormAndResidualMLP/mlp/linear_1/MatMul_3:product:0ALayerNormAndResidualMLP/mlp/linear_1/Add_3/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2,
*LayerNormAndResidualMLP/mlp/linear_1/Add_3п
6LayerNormAndResidualMLP/ResidualLayernormWrapper/add_3AddV2.LayerNormAndResidualMLP/mlp/linear_1/Add_3:z:0QLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/add_1:z:0*
T0*(
_output_shapes
:         А28
6LayerNormAndResidualMLP/ResidualLayernormWrapper/add_3П
\LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_3/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
         2^
\LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_3/mean/reduction_indicesЖ
JLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_3/meanMean:LayerNormAndResidualMLP/ResidualLayernormWrapper/add_3:z:0eLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_3/mean/reduction_indices:output:0*
T0*'
_output_shapes
:         *
	keep_dims(2L
JLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_3/mean┐
RLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_3/StopGradientStopGradientSLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_3/mean:output:0*
T0*'
_output_shapes
:         2T
RLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_3/StopGradientУ
WLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_3/SquaredDifferenceSquaredDifference:LayerNormAndResidualMLP/ResidualLayernormWrapper/add_3:z:0[LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_3/StopGradient:output:0*
T0*(
_output_shapes
:         А2Y
WLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_3/SquaredDifferenceЧ
`LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_3/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
         2b
`LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_3/variance/reduction_indices│
NLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_3/varianceMean[LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_3/SquaredDifference:z:0iLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_3/variance/reduction_indices:output:0*
T0*'
_output_shapes
:         *
	keep_dims(2P
NLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_3/varianceу
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *м┼'72O
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/add/yЖ
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/addAddV2WLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_3/variance:output:0VLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/add/y:output:0*
T0*'
_output_shapes
:         2M
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/addк
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/RsqrtRsqrtOLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/add:z:0*
T0*'
_output_shapes
:         2O
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/Rsqrt┘
ZLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/mul/ReadVariableOpReadVariableOpclayernormandresidualmlp_residuallayernormwrapper_layer_norm_batchnorm_3_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02\
ZLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/mul/ReadVariableOpЛ
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/mulMulQLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/Rsqrt:y:0bLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/mul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2M
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/mulх
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/mul_1Mul:LayerNormAndResidualMLP/ResidualLayernormWrapper/add_3:z:0OLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/mul:z:0*
T0*(
_output_shapes
:         А2O
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/mul_1■
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/mul_2MulSLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_3/mean:output:0OLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/mul:z:0*
T0*(
_output_shapes
:         А2O
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/mul_2ш
_LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/Identity/ReadVariableOpReadVariableOphlayernormandresidualmlp_residuallayernormwrapper_layer_norm_batchnorm_3_identity_readvariableop_resource*
_output_shapes	
:А*
dtype02a
_LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/Identity/ReadVariableOp┐
PLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/IdentityIdentitygLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/Identity/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2R
PLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/IdentityВ
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/subSubYLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/Identity:output:0QLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/mul_2:z:0*
T0*(
_output_shapes
:         А2M
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/sub■
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/add_1AddV2QLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/mul_1:z:0OLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/sub:z:0*
T0*(
_output_shapes
:         А2O
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/add_1Й
 MultivariateNormalDiagHead/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 MultivariateNormalDiagHead/ConstЇ
7MultivariateNormalDiagHead/linear/MatMul/ReadVariableOpReadVariableOp@multivariatenormaldiaghead_linear_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype029
7MultivariateNormalDiagHead/linear/MatMul/ReadVariableOpд
(MultivariateNormalDiagHead/linear/MatMulMatMulQLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/add_1:z:0?MultivariateNormalDiagHead/linear/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2*
(MultivariateNormalDiagHead/linear/MatMulц
4MultivariateNormalDiagHead/linear/Add/ReadVariableOpReadVariableOp=multivariatenormaldiaghead_linear_add_readvariableop_resource*
_output_shapes
:*
dtype026
4MultivariateNormalDiagHead/linear/Add/ReadVariableOp∙
%MultivariateNormalDiagHead/linear/AddAdd2MultivariateNormalDiagHead/linear/MatMul:product:0<MultivariateNormalDiagHead/linear/Add/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2'
%MultivariateNormalDiagHead/linear/Add·
9MultivariateNormalDiagHead/linear/MatMul_1/ReadVariableOpReadVariableOpBmultivariatenormaldiaghead_linear_matmul_1_readvariableop_resource*
_output_shapes
:	А*
dtype02;
9MultivariateNormalDiagHead/linear/MatMul_1/ReadVariableOpк
*MultivariateNormalDiagHead/linear/MatMul_1MatMulQLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/add_1:z:0AMultivariateNormalDiagHead/linear/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2,
*MultivariateNormalDiagHead/linear/MatMul_1ь
6MultivariateNormalDiagHead/linear/Add_1/ReadVariableOpReadVariableOp?multivariatenormaldiaghead_linear_add_1_readvariableop_resource*
_output_shapes
:*
dtype028
6MultivariateNormalDiagHead/linear/Add_1/ReadVariableOpБ
'MultivariateNormalDiagHead/linear/Add_1Add4MultivariateNormalDiagHead/linear/MatMul_1:product:0>MultivariateNormalDiagHead/linear/Add_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2)
'MultivariateNormalDiagHead/linear/Add_1╡
#MultivariateNormalDiagHead/SoftplusSoftplus+MultivariateNormalDiagHead/linear/Add_1:z:0*
T0*'
_output_shapes
:         2%
#MultivariateNormalDiagHead/Softplusж
%MultivariateNormalDiagHead/Softplus_1Softplus)MultivariateNormalDiagHead/Const:output:0*
T0*
_output_shapes
: 2'
%MultivariateNormalDiagHead/Softplus_1С
$MultivariateNormalDiagHead/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2&
$MultivariateNormalDiagHead/truediv/x╪
"MultivariateNormalDiagHead/truedivRealDiv-MultivariateNormalDiagHead/truediv/x:output:03MultivariateNormalDiagHead/Softplus_1:activations:0*
T0*
_output_shapes
: 2$
"MultivariateNormalDiagHead/truediv╘
MultivariateNormalDiagHead/mulMul1MultivariateNormalDiagHead/Softplus:activations:0&MultivariateNormalDiagHead/truediv:z:0*
T0*'
_output_shapes
:         2 
MultivariateNormalDiagHead/mulЙ
 MultivariateNormalDiagHead/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж52"
 MultivariateNormalDiagHead/add/y╩
MultivariateNormalDiagHead/addAddV2"MultivariateNormalDiagHead/mul:z:0)MultivariateNormalDiagHead/add/y:output:0*
T0*'
_output_shapes
:         2 
MultivariateNormalDiagHead/add╝
{MultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/range_dimension_tensor/ConstConst*
_output_shapes
: *
dtype0*
value	B :2}
{MultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/range_dimension_tensor/Const№
WMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/event_shapeConst*
_output_shapes
:*
dtype0*
valueB:2Y
WMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/event_shapeЗ
ЧMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/ShapeShape"MultivariateNormalDiagHead/add:z:0*
T0*
_output_shapes
:2Ъ
ЧMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/Shapeд
еMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2и
еMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stackЯ
зMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2к
зMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_1Я
зMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2к
зMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_2╒
ЯMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_sliceStridedSliceаMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/Shape:output:0оMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack:output:0░MultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_1:output:0░MultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2в
ЯMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_sliceк
бMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/values_1PackиMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice:output:0*
N*
T0*
_output_shapes
:2д
бMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/values_1Г
ЭMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2а
ЭMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/axisъ
ШMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concatConcatV2аMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/Shape:output:0кMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/values_1:output:0жMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/axis:output:0*
N*
T0*
_output_shapes
:2Ы
ШMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat█
ЕMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2И
ЕMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stackш
ЗMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
■        2К
ЗMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_1▀
ЗMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2К
ЗMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_2▓
MultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_sliceStridedSliceбMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat:output:0ОMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack:output:0РMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_1:output:0РMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2Б
MultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice 
QMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/ShapeShape)MultivariateNormalDiagHead/linear/Add:z:0*
T0*
_output_shapes
:2S
QMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/ShapeМ
_MultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2a
_MultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stackЩ
aMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         2c
aMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_1Р
aMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2c
aMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_2и
YMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_sliceStridedSliceZMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/Shape:output:0hMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack:output:0jMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_1:output:0jMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2[
YMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_sliceО
wMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/prefer_static_broadcast_shape/BroadcastArgsBroadcastArgsИMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice:output:0bMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice:output:0*
_output_shapes
:2y
wMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/prefer_static_broadcast_shape/BroadcastArgs├
=MultivariateNormalDiagHead/MultivariateNormalDiag/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2?
=MultivariateNormalDiagHead/MultivariateNormalDiag/zeros/Constю
7MultivariateNormalDiagHead/MultivariateNormalDiag/zerosFill|MultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/prefer_static_broadcast_shape/BroadcastArgs:r0:0FMultivariateNormalDiagHead/MultivariateNormalDiag/zeros/Const:output:0*
T0*#
_output_shapes
:         29
7MultivariateNormalDiagHead/MultivariateNormalDiag/zeros╡
6MultivariateNormalDiagHead/MultivariateNormalDiag/onesConst*
_output_shapes
: *
dtype0*
valueB
 *  А?28
6MultivariateNormalDiagHead/MultivariateNormalDiag/ones▓
6MultivariateNormalDiagHead/MultivariateNormalDiag/zeroConst*
_output_shapes
: *
dtype0*
value	B : 28
6MultivariateNormalDiagHead/MultivariateNormalDiag/zero╡
7MultivariateNormalDiagHead/MultivariateNormalDiag/emptyConst*
_output_shapes
: *
dtype0*
valueB 29
7MultivariateNormalDiagHead/MultivariateNormalDiag/emptyГ
^stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB 2`
^stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/sample_shapeН
тstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal/is_scalar_batch/is_scalar_batchConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2х
тstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal/is_scalar_batch/is_scalar_batchК
bstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/pick_vector/condConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2d
bstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/pick_vector/condв
jstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/pick_vector/false_vectorConst*
_output_shapes
:*
dtype0*
valueB:2l
jstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/pick_vector/false_vectorО
dstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/pick_vector_1/condConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2f
dstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/pick_vector_1/condд
kstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/pick_vector_1/true_vectorConst*
_output_shapes
:*
dtype0*
valueB:2m
kstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/pick_vector_1/true_vectorЕ
┌stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/sample_shape/xConst*
_output_shapes
:*
dtype0*
valueB:2▌
┌stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/sample_shape/xМ
Ъstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/sample_shape/xConst*
_output_shapes
:*
dtype0*
valueB"      2Э
Ъstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/sample_shape/xЩ
Сstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/ShapeShape@MultivariateNormalDiagHead/MultivariateNormalDiag/zeros:output:0*
T0*
_output_shapes
:2Ф
Сstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/ShapeВ
Ьstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/BroadcastArgs/s1Const*
_output_shapes
: *
dtype0*
valueB 2Я
Ьstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/BroadcastArgs/s1л	
Щstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/BroadcastArgsBroadcastArgsЪstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/Shape:output:0еstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/BroadcastArgs/s1:output:0*
_output_shapes
:2Ь
Щstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/BroadcastArgsЗ
Ыstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:2Ю
Ыstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/concat/values_0ў
Чstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2Ъ
Чstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/concat/axis╨
Тstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/concatConcatV2дstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/concat/values_0:output:0Юstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/BroadcastArgs:r0:0аstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2Х
Тstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/concatЦ
еstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2и
еstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/normal/random_normal/meanЪ
зstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2к
зstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/normal/random_normal/stddevц
╡stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormalЫstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/concat:output:0*
T0*'
_output_shapes
:         *
dtype02╕
╡stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/normal/random_normal/RandomStandardNormal№	
дstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/normal/random_normal/mulMul╛stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/normal/random_normal/RandomStandardNormal:output:0░stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/normal/random_normal/stddev:output:0*
T0*'
_output_shapes
:         2з
дstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/normal/random_normal/mul▄	
аstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/normal/random_normalAddиstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/normal/random_normal/mul:z:0оstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/normal/random_normal/mean:output:0*
T0*'
_output_shapes
:         2г
аstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/normal/random_normal╞
Пstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/mulMulдstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/normal/random_normal:z:0?MultivariateNormalDiagHead/MultivariateNormalDiag/ones:output:0*
T0*'
_output_shapes
:         2Т
Пstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/mul╕
Пstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/addAddV2Уstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/mul:z:0@MultivariateNormalDiagHead/MultivariateNormalDiag/zeros:output:0*
T0*'
_output_shapes
:         2Т
Пstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/addё
Уstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/Shape_1ShapeУstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/add:z:0*
T0*
_output_shapes
:2Ц
Уstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/Shape_1П
Яstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2в
Яstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/strided_slice/stackУ
бstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2д
бstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/strided_slice/stack_1У
бstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2д
бstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/strided_slice/stack_2п
Щstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/strided_sliceStridedSliceЬstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/Shape_1:output:0иstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/strided_slice/stack:output:0кstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/strided_slice/stack_1:output:0кstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2Ь
Щstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/strided_slice√
Щstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2Ь
Щstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/concat_1/axis┘
Фstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/concat_1ConcatV2гstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/sample_shape/x:output:0вstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/strided_slice:output:0вstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2Ч
Фstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/concat_1д	
Уstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/ReshapeReshapeУstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/add:z:0Эstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/concat_1:output:0*
T0*+
_output_shapes
:         2Ц
Уstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/ReshapeР
┌stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2▌
┌stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/transpose/perm∙
╒stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/transpose	TransposeЬstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/Reshape:output:0уstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/transpose/perm:output:0*
T0*+
_output_shapes
:         2╪
╒stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/transpose│
╤stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/ShapeShape┘stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/transpose:y:0*
T0*
_output_shapes
:2╘
╤stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/ShapeП
▀stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2т
▀stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/strided_slice/stackУ
сstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2ф
сstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/strided_slice/stack_1У
сstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2ф
сstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/strided_slice/stack_2н
┘stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/strided_sliceStridedSlice┌stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/Shape:output:0шstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/strided_slice/stack:output:0ъstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/strided_slice/stack_1:output:0ъstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2▄
┘stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/strided_sliceў
╫stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2┌
╫stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/concat/axisУ	
╥stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/concatConcatV2уstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/sample_shape/x:output:0тstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/strided_slice:output:0рstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2╒
╥stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/concatи
╙stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/ReshapeReshape┘stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/transpose:y:0█stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/concat:output:0*
T0*+
_output_shapes
:         2╓
╙stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/Reshape┐
Wstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/ShapeShape▄stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/Reshape:output:0*
T0*
_output_shapes
:2Y
Wstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/ShapeШ
estochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2g
estochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/strided_slice/stackЬ
gstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2i
gstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/strided_slice/stack_1Ь
gstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2i
gstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/strided_slice/stack_2╩
_stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/strided_sliceStridedSlice`stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/Shape:output:0nstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/strided_slice/stack:output:0pstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/strided_slice/stack_1:output:0pstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2a
_stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/strided_sliceА
]stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2_
]stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/concat/axisй
Xstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/concatConcatV2gstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/sample_shape:output:0hstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/strided_slice:output:0fstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2Z
Xstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/concat╡
Ystochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/ReshapeReshape▄stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/Reshape:output:0astochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/concat:output:0*
T0*'
_output_shapes
:         2[
Ystochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/Reshapeю
Уstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_chain_of_MultivariateNormalDiagHead_MultivariateNormalDiag_shift_of_MultivariateNormalDiagHead_MultivariateNormalDiag_scale_matvec_linear_operator/forward/MultivariateNormalDiagHead_MultivariateNormalDiag_scale_matvec_linear_operator/forward/LinearOperatorDiag/matvec/mulMul"MultivariateNormalDiagHead/add:z:0bstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/Reshape:output:0*
T0*'
_output_shapes
:         2Ц
Уstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_chain_of_MultivariateNormalDiagHead_MultivariateNormalDiag_shift_of_MultivariateNormalDiagHead_MultivariateNormalDiag_scale_matvec_linear_operator/forward/MultivariateNormalDiagHead_MultivariateNormalDiag_scale_matvec_linear_operator/forward/LinearOperatorDiag/matvec/mul╦	
тstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_chain_of_MultivariateNormalDiagHead_MultivariateNormalDiag_shift_of_MultivariateNormalDiagHead_MultivariateNormalDiag_scale_matvec_linear_operator/forward/MultivariateNormalDiagHead_MultivariateNormalDiag_shift/forward/addAddV2Чstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_chain_of_MultivariateNormalDiagHead_MultivariateNormalDiag_shift_of_MultivariateNormalDiagHead_MultivariateNormalDiag_scale_matvec_linear_operator/forward/MultivariateNormalDiagHead_MultivariateNormalDiag_scale_matvec_linear_operator/forward/LinearOperatorDiag/matvec/mul:z:0)MultivariateNormalDiagHead/linear/Add:z:0*
T0*'
_output_shapes
:         2х
тstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_chain_of_MultivariateNormalDiagHead_MultivariateNormalDiag_shift_of_MultivariateNormalDiagHead_MultivariateNormalDiag_scale_matvec_linear_operator/forward/MultivariateNormalDiagHead_MultivariateNormalDiag_shift/forward/add╗
IdentityIdentityцstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_chain_of_MultivariateNormalDiagHead_MultivariateNormalDiag_shift_of_MultivariateNormalDiagHead_MultivariateNormalDiag_scale_matvec_linear_operator/forward/MultivariateNormalDiagHead_MultivariateNormalDiag_shift/forward/add:z:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*щ
_input_shapes╫
╘:         :         :         :         :::::::::::::::::::::::::::::::::::O K
'
_output_shapes
:         
 
_user_specified_nameargs_0:OK
'
_output_shapes
:         
 
_user_specified_nameargs_0:OK
'
_output_shapes
:         
 
_user_specified_nameargs_0:OK
'
_output_shapes
:         
 
_user_specified_nameargs_0:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	
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
: :
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: 
ян
г
 __inference_wrapped_module_82517
args_0_joint_angles
args_0_target
args_0_upright
args_0_velocity<
8control_network_normalize_matmul_readvariableop_resource9
5control_network_normalize_add_readvariableop_resourceD
@control_network_layer_norm_batchnorm_mul_readvariableop_resourceI
Econtrol_network_layer_norm_batchnorm_identity_readvariableop_resourceG
Clayernormandresidualmlp_mlp_linear_0_matmul_readvariableop_resourceD
@layernormandresidualmlp_mlp_linear_0_add_readvariableop_resourceI
Elayernormandresidualmlp_mlp_linear_0_matmul_1_readvariableop_resourceF
Blayernormandresidualmlp_mlp_linear_0_add_1_readvariableop_resourceG
Clayernormandresidualmlp_mlp_linear_1_matmul_readvariableop_resourceD
@layernormandresidualmlp_mlp_linear_1_add_readvariableop_resourcee
alayernormandresidualmlp_residuallayernormwrapper_layer_norm_batchnorm_mul_readvariableop_resourcej
flayernormandresidualmlp_residuallayernormwrapper_layer_norm_batchnorm_identity_readvariableop_resourceI
Elayernormandresidualmlp_mlp_linear_0_matmul_2_readvariableop_resourceF
Blayernormandresidualmlp_mlp_linear_0_add_2_readvariableop_resourceI
Elayernormandresidualmlp_mlp_linear_1_matmul_1_readvariableop_resourceF
Blayernormandresidualmlp_mlp_linear_1_add_1_readvariableop_resourceg
clayernormandresidualmlp_residuallayernormwrapper_layer_norm_batchnorm_1_mul_readvariableop_resourcel
hlayernormandresidualmlp_residuallayernormwrapper_layer_norm_batchnorm_1_identity_readvariableop_resourceI
Elayernormandresidualmlp_mlp_linear_0_matmul_3_readvariableop_resourceF
Blayernormandresidualmlp_mlp_linear_0_add_3_readvariableop_resourceI
Elayernormandresidualmlp_mlp_linear_1_matmul_2_readvariableop_resourceF
Blayernormandresidualmlp_mlp_linear_1_add_2_readvariableop_resourceg
clayernormandresidualmlp_residuallayernormwrapper_layer_norm_batchnorm_2_mul_readvariableop_resourcel
hlayernormandresidualmlp_residuallayernormwrapper_layer_norm_batchnorm_2_identity_readvariableop_resourceI
Elayernormandresidualmlp_mlp_linear_0_matmul_4_readvariableop_resourceF
Blayernormandresidualmlp_mlp_linear_0_add_4_readvariableop_resourceI
Elayernormandresidualmlp_mlp_linear_1_matmul_3_readvariableop_resourceF
Blayernormandresidualmlp_mlp_linear_1_add_3_readvariableop_resourceg
clayernormandresidualmlp_residuallayernormwrapper_layer_norm_batchnorm_3_mul_readvariableop_resourcel
hlayernormandresidualmlp_residuallayernormwrapper_layer_norm_batchnorm_3_identity_readvariableop_resourceD
@multivariatenormaldiaghead_linear_matmul_readvariableop_resourceA
=multivariatenormaldiaghead_linear_add_readvariableop_resourceF
Bmultivariatenormaldiaghead_linear_matmul_1_readvariableop_resourceC
?multivariatenormaldiaghead_linear_add_1_readvariableop_resource
identityИБ
control_network/flatten/ShapeShapeargs_0_joint_angles*
T0*
_output_shapes
:2
control_network/flatten/Shapeд
+control_network/flatten/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+control_network/flatten/strided_slice/stackи
-control_network/flatten/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-control_network/flatten/strided_slice/stack_1и
-control_network/flatten/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-control_network/flatten/strided_slice/stack_2Ё
%control_network/flatten/strided_sliceStridedSlice&control_network/flatten/Shape:output:04control_network/flatten/strided_slice/stack:output:06control_network/flatten/strided_slice/stack_1:output:06control_network/flatten/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2'
%control_network/flatten/strided_sliceЬ
'control_network/flatten/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'control_network/flatten/concat/values_1М
#control_network/flatten/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#control_network/flatten/concat/axisК
control_network/flatten/concatConcatV2.control_network/flatten/strided_slice:output:00control_network/flatten/concat/values_1:output:0,control_network/flatten/concat/axis:output:0*
N*
T0*
_output_shapes
:2 
control_network/flatten/concat╜
control_network/flatten/ReshapeReshapeargs_0_joint_angles'control_network/flatten/concat:output:0*
T0*'
_output_shapes
:         2!
control_network/flatten/Reshape
control_network/flatten_1/ShapeShapeargs_0_target*
T0*
_output_shapes
:2!
control_network/flatten_1/Shapeи
-control_network/flatten_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-control_network/flatten_1/strided_slice/stackм
/control_network/flatten_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/control_network/flatten_1/strided_slice/stack_1м
/control_network/flatten_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/control_network/flatten_1/strided_slice/stack_2№
'control_network/flatten_1/strided_sliceStridedSlice(control_network/flatten_1/Shape:output:06control_network/flatten_1/strided_slice/stack:output:08control_network/flatten_1/strided_slice/stack_1:output:08control_network/flatten_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2)
'control_network/flatten_1/strided_sliceа
)control_network/flatten_1/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)control_network/flatten_1/concat/values_1Р
%control_network/flatten_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2'
%control_network/flatten_1/concat/axisФ
 control_network/flatten_1/concatConcatV20control_network/flatten_1/strided_slice:output:02control_network/flatten_1/concat/values_1:output:0.control_network/flatten_1/concat/axis:output:0*
N*
T0*
_output_shapes
:2"
 control_network/flatten_1/concat╜
!control_network/flatten_1/ReshapeReshapeargs_0_target)control_network/flatten_1/concat:output:0*
T0*'
_output_shapes
:         2#
!control_network/flatten_1/ReshapeА
control_network/flatten_2/ShapeShapeargs_0_upright*
T0*
_output_shapes
:2!
control_network/flatten_2/Shapeи
-control_network/flatten_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-control_network/flatten_2/strided_slice/stackм
/control_network/flatten_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/control_network/flatten_2/strided_slice/stack_1м
/control_network/flatten_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/control_network/flatten_2/strided_slice/stack_2№
'control_network/flatten_2/strided_sliceStridedSlice(control_network/flatten_2/Shape:output:06control_network/flatten_2/strided_slice/stack:output:08control_network/flatten_2/strided_slice/stack_1:output:08control_network/flatten_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2)
'control_network/flatten_2/strided_sliceа
)control_network/flatten_2/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)control_network/flatten_2/concat/values_1Р
%control_network/flatten_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2'
%control_network/flatten_2/concat/axisФ
 control_network/flatten_2/concatConcatV20control_network/flatten_2/strided_slice:output:02control_network/flatten_2/concat/values_1:output:0.control_network/flatten_2/concat/axis:output:0*
N*
T0*
_output_shapes
:2"
 control_network/flatten_2/concat╛
!control_network/flatten_2/ReshapeReshapeargs_0_upright)control_network/flatten_2/concat:output:0*
T0*'
_output_shapes
:         2#
!control_network/flatten_2/ReshapeБ
control_network/flatten_3/ShapeShapeargs_0_velocity*
T0*
_output_shapes
:2!
control_network/flatten_3/Shapeи
-control_network/flatten_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-control_network/flatten_3/strided_slice/stackм
/control_network/flatten_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/control_network/flatten_3/strided_slice/stack_1м
/control_network/flatten_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/control_network/flatten_3/strided_slice/stack_2№
'control_network/flatten_3/strided_sliceStridedSlice(control_network/flatten_3/Shape:output:06control_network/flatten_3/strided_slice/stack:output:08control_network/flatten_3/strided_slice/stack_1:output:08control_network/flatten_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2)
'control_network/flatten_3/strided_sliceа
)control_network/flatten_3/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)control_network/flatten_3/concat/values_1Р
%control_network/flatten_3/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2'
%control_network/flatten_3/concat/axisФ
 control_network/flatten_3/concatConcatV20control_network/flatten_3/strided_slice:output:02control_network/flatten_3/concat/values_1:output:0.control_network/flatten_3/concat/axis:output:0*
N*
T0*
_output_shapes
:2"
 control_network/flatten_3/concat┐
!control_network/flatten_3/ReshapeReshapeargs_0_velocity)control_network/flatten_3/concat:output:0*
T0*'
_output_shapes
:         2#
!control_network/flatten_3/Reshape|
control_network/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
control_network/concat/axis╦
control_network/concatConcatV2(control_network/flatten/Reshape:output:0*control_network/flatten_1/Reshape:output:0*control_network/flatten_2/Reshape:output:0*control_network/flatten_3/Reshape:output:0$control_network/concat/axis:output:0*
N*
T0*'
_output_shapes
:         2
control_network/concat▄
/control_network/normalize/MatMul/ReadVariableOpReadVariableOp8control_network_normalize_matmul_readvariableop_resource*
_output_shapes
:	м*
dtype021
/control_network/normalize/MatMul/ReadVariableOp█
 control_network/normalize/MatMulMatMulcontrol_network/concat:output:07control_network/normalize/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         м2"
 control_network/normalize/MatMul╧
,control_network/normalize/Add/ReadVariableOpReadVariableOp5control_network_normalize_add_readvariableop_resource*
_output_shapes	
:м*
dtype02.
,control_network/normalize/Add/ReadVariableOp┌
control_network/normalize/AddAdd*control_network/normalize/MatMul:product:04control_network/normalize/Add/ReadVariableOp:value:0*
T0*(
_output_shapes
:         м2
control_network/normalize/Add└
9control_network/layer_norm/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9control_network/layer_norm/moments/mean/reduction_indicesД
'control_network/layer_norm/moments/meanMean!control_network/normalize/Add:z:0Bcontrol_network/layer_norm/moments/mean/reduction_indices:output:0*
T0*'
_output_shapes
:         *
	keep_dims(2)
'control_network/layer_norm/moments/mean╓
/control_network/layer_norm/moments/StopGradientStopGradient0control_network/layer_norm/moments/mean:output:0*
T0*'
_output_shapes
:         21
/control_network/layer_norm/moments/StopGradientС
4control_network/layer_norm/moments/SquaredDifferenceSquaredDifference!control_network/normalize/Add:z:08control_network/layer_norm/moments/StopGradient:output:0*
T0*(
_output_shapes
:         м26
4control_network/layer_norm/moments/SquaredDifference╚
=control_network/layer_norm/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2?
=control_network/layer_norm/moments/variance/reduction_indicesз
+control_network/layer_norm/moments/varianceMean8control_network/layer_norm/moments/SquaredDifference:z:0Fcontrol_network/layer_norm/moments/variance/reduction_indices:output:0*
T0*'
_output_shapes
:         *
	keep_dims(2-
+control_network/layer_norm/moments/varianceЭ
*control_network/layer_norm/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *м┼'72,
*control_network/layer_norm/batchnorm/add/y·
(control_network/layer_norm/batchnorm/addAddV24control_network/layer_norm/moments/variance:output:03control_network/layer_norm/batchnorm/add/y:output:0*
T0*'
_output_shapes
:         2*
(control_network/layer_norm/batchnorm/add┴
*control_network/layer_norm/batchnorm/RsqrtRsqrt,control_network/layer_norm/batchnorm/add:z:0*
T0*'
_output_shapes
:         2,
*control_network/layer_norm/batchnorm/RsqrtЁ
7control_network/layer_norm/batchnorm/mul/ReadVariableOpReadVariableOp@control_network_layer_norm_batchnorm_mul_readvariableop_resource*
_output_shapes	
:м*
dtype029
7control_network/layer_norm/batchnorm/mul/ReadVariableOp 
(control_network/layer_norm/batchnorm/mulMul.control_network/layer_norm/batchnorm/Rsqrt:y:0?control_network/layer_norm/batchnorm/mul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         м2*
(control_network/layer_norm/batchnorm/mulу
*control_network/layer_norm/batchnorm/mul_1Mul!control_network/normalize/Add:z:0,control_network/layer_norm/batchnorm/mul:z:0*
T0*(
_output_shapes
:         м2,
*control_network/layer_norm/batchnorm/mul_1Є
*control_network/layer_norm/batchnorm/mul_2Mul0control_network/layer_norm/moments/mean:output:0,control_network/layer_norm/batchnorm/mul:z:0*
T0*(
_output_shapes
:         м2,
*control_network/layer_norm/batchnorm/mul_2 
<control_network/layer_norm/batchnorm/Identity/ReadVariableOpReadVariableOpEcontrol_network_layer_norm_batchnorm_identity_readvariableop_resource*
_output_shapes	
:м*
dtype02>
<control_network/layer_norm/batchnorm/Identity/ReadVariableOp╓
-control_network/layer_norm/batchnorm/IdentityIdentityDcontrol_network/layer_norm/batchnorm/Identity/ReadVariableOp:value:0*
T0*
_output_shapes	
:м2/
-control_network/layer_norm/batchnorm/IdentityЎ
(control_network/layer_norm/batchnorm/subSub6control_network/layer_norm/batchnorm/Identity:output:0.control_network/layer_norm/batchnorm/mul_2:z:0*
T0*(
_output_shapes
:         м2*
(control_network/layer_norm/batchnorm/subЄ
*control_network/layer_norm/batchnorm/add_1AddV2.control_network/layer_norm/batchnorm/mul_1:z:0,control_network/layer_norm/batchnorm/sub:z:0*
T0*(
_output_shapes
:         м2,
*control_network/layer_norm/batchnorm/add_1Ч
control_network/TanhTanh.control_network/layer_norm/batchnorm/add_1:z:0*
T0*(
_output_shapes
:         м2
control_network/TanhМ
#control_network/concat_1/concat_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#control_network/concat_1/concat_dimЫ
control_network/concat_1/concatIdentitycontrol_network/Tanh:y:0*
T0*(
_output_shapes
:         м2!
control_network/concat_1/concat■
:LayerNormAndResidualMLP/mlp/linear_0/MatMul/ReadVariableOpReadVariableOpClayernormandresidualmlp_mlp_linear_0_matmul_readvariableop_resource* 
_output_shapes
:
мА*
dtype02<
:LayerNormAndResidualMLP/mlp/linear_0/MatMul/ReadVariableOpЕ
+LayerNormAndResidualMLP/mlp/linear_0/MatMulMatMul(control_network/concat_1/concat:output:0BLayerNormAndResidualMLP/mlp/linear_0/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2-
+LayerNormAndResidualMLP/mlp/linear_0/MatMulЁ
7LayerNormAndResidualMLP/mlp/linear_0/Add/ReadVariableOpReadVariableOp@layernormandresidualmlp_mlp_linear_0_add_readvariableop_resource*
_output_shapes	
:А*
dtype029
7LayerNormAndResidualMLP/mlp/linear_0/Add/ReadVariableOpЖ
(LayerNormAndResidualMLP/mlp/linear_0/AddAdd5LayerNormAndResidualMLP/mlp/linear_0/MatMul:product:0?LayerNormAndResidualMLP/mlp/linear_0/Add/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2*
(LayerNormAndResidualMLP/mlp/linear_0/AddД
<LayerNormAndResidualMLP/mlp/linear_0/MatMul_1/ReadVariableOpReadVariableOpElayernormandresidualmlp_mlp_linear_0_matmul_1_readvariableop_resource* 
_output_shapes
:
АА*
dtype02>
<LayerNormAndResidualMLP/mlp/linear_0/MatMul_1/ReadVariableOpП
-LayerNormAndResidualMLP/mlp/linear_0/MatMul_1MatMul,LayerNormAndResidualMLP/mlp/linear_0/Add:z:0DLayerNormAndResidualMLP/mlp/linear_0/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2/
-LayerNormAndResidualMLP/mlp/linear_0/MatMul_1Ў
9LayerNormAndResidualMLP/mlp/linear_0/Add_1/ReadVariableOpReadVariableOpBlayernormandresidualmlp_mlp_linear_0_add_1_readvariableop_resource*
_output_shapes	
:А*
dtype02;
9LayerNormAndResidualMLP/mlp/linear_0/Add_1/ReadVariableOpО
*LayerNormAndResidualMLP/mlp/linear_0/Add_1Add7LayerNormAndResidualMLP/mlp/linear_0/MatMul_1:product:0ALayerNormAndResidualMLP/mlp/linear_0/Add_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2,
*LayerNormAndResidualMLP/mlp/linear_0/Add_1п
 LayerNormAndResidualMLP/mlp/ReluRelu.LayerNormAndResidualMLP/mlp/linear_0/Add_1:z:0*
T0*(
_output_shapes
:         А2"
 LayerNormAndResidualMLP/mlp/Relu■
:LayerNormAndResidualMLP/mlp/linear_1/MatMul/ReadVariableOpReadVariableOpClayernormandresidualmlp_mlp_linear_1_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype02<
:LayerNormAndResidualMLP/mlp/linear_1/MatMul/ReadVariableOpЛ
+LayerNormAndResidualMLP/mlp/linear_1/MatMulMatMul.LayerNormAndResidualMLP/mlp/Relu:activations:0BLayerNormAndResidualMLP/mlp/linear_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2-
+LayerNormAndResidualMLP/mlp/linear_1/MatMulЁ
7LayerNormAndResidualMLP/mlp/linear_1/Add/ReadVariableOpReadVariableOp@layernormandresidualmlp_mlp_linear_1_add_readvariableop_resource*
_output_shapes	
:А*
dtype029
7LayerNormAndResidualMLP/mlp/linear_1/Add/ReadVariableOpЖ
(LayerNormAndResidualMLP/mlp/linear_1/AddAdd5LayerNormAndResidualMLP/mlp/linear_1/MatMul:product:0?LayerNormAndResidualMLP/mlp/linear_1/Add/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2*
(LayerNormAndResidualMLP/mlp/linear_1/AddД
4LayerNormAndResidualMLP/ResidualLayernormWrapper/addAddV2,LayerNormAndResidualMLP/mlp/linear_1/Add:z:0,LayerNormAndResidualMLP/mlp/linear_0/Add:z:0*
T0*(
_output_shapes
:         А26
4LayerNormAndResidualMLP/ResidualLayernormWrapper/addЛ
ZLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
         2\
ZLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments/mean/reduction_indices■
HLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments/meanMean8LayerNormAndResidualMLP/ResidualLayernormWrapper/add:z:0cLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments/mean/reduction_indices:output:0*
T0*'
_output_shapes
:         *
	keep_dims(2J
HLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments/mean╣
PLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments/StopGradientStopGradientQLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments/mean:output:0*
T0*'
_output_shapes
:         2R
PLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments/StopGradientЛ
ULayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments/SquaredDifferenceSquaredDifference8LayerNormAndResidualMLP/ResidualLayernormWrapper/add:z:0YLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments/StopGradient:output:0*
T0*(
_output_shapes
:         А2W
ULayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments/SquaredDifferenceУ
^LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
         2`
^LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments/variance/reduction_indicesл
LLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments/varianceMeanYLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments/SquaredDifference:z:0gLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments/variance/reduction_indices:output:0*
T0*'
_output_shapes
:         *
	keep_dims(2N
LLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments/variance▀
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *м┼'72M
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/add/y■
ILayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/addAddV2ULayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments/variance:output:0TLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/add/y:output:0*
T0*'
_output_shapes
:         2K
ILayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/addд
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/RsqrtRsqrtMLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/add:z:0*
T0*'
_output_shapes
:         2M
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/Rsqrt╙
XLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/mul/ReadVariableOpReadVariableOpalayernormandresidualmlp_residuallayernormwrapper_layer_norm_batchnorm_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02Z
XLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/mul/ReadVariableOpГ
ILayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/mulMulOLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/Rsqrt:y:0`LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/mul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2K
ILayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/mul▌
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/mul_1Mul8LayerNormAndResidualMLP/ResidualLayernormWrapper/add:z:0MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/mul:z:0*
T0*(
_output_shapes
:         А2M
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/mul_1Ў
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/mul_2MulQLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments/mean:output:0MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/mul:z:0*
T0*(
_output_shapes
:         А2M
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/mul_2т
]LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/Identity/ReadVariableOpReadVariableOpflayernormandresidualmlp_residuallayernormwrapper_layer_norm_batchnorm_identity_readvariableop_resource*
_output_shapes	
:А*
dtype02_
]LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/Identity/ReadVariableOp╣
NLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/IdentityIdentityeLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/Identity/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2P
NLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/Identity·
ILayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/subSubWLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/Identity:output:0OLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/mul_2:z:0*
T0*(
_output_shapes
:         А2K
ILayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/subЎ
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/add_1AddV2OLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/mul_1:z:0MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/sub:z:0*
T0*(
_output_shapes
:         А2M
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/add_1Д
<LayerNormAndResidualMLP/mlp/linear_0/MatMul_2/ReadVariableOpReadVariableOpElayernormandresidualmlp_mlp_linear_0_matmul_2_readvariableop_resource* 
_output_shapes
:
АА*
dtype02>
<LayerNormAndResidualMLP/mlp/linear_0/MatMul_2/ReadVariableOp▓
-LayerNormAndResidualMLP/mlp/linear_0/MatMul_2MatMulOLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/add_1:z:0DLayerNormAndResidualMLP/mlp/linear_0/MatMul_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2/
-LayerNormAndResidualMLP/mlp/linear_0/MatMul_2Ў
9LayerNormAndResidualMLP/mlp/linear_0/Add_2/ReadVariableOpReadVariableOpBlayernormandresidualmlp_mlp_linear_0_add_2_readvariableop_resource*
_output_shapes	
:А*
dtype02;
9LayerNormAndResidualMLP/mlp/linear_0/Add_2/ReadVariableOpО
*LayerNormAndResidualMLP/mlp/linear_0/Add_2Add7LayerNormAndResidualMLP/mlp/linear_0/MatMul_2:product:0ALayerNormAndResidualMLP/mlp/linear_0/Add_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2,
*LayerNormAndResidualMLP/mlp/linear_0/Add_2│
"LayerNormAndResidualMLP/mlp/Relu_1Relu.LayerNormAndResidualMLP/mlp/linear_0/Add_2:z:0*
T0*(
_output_shapes
:         А2$
"LayerNormAndResidualMLP/mlp/Relu_1Д
<LayerNormAndResidualMLP/mlp/linear_1/MatMul_1/ReadVariableOpReadVariableOpElayernormandresidualmlp_mlp_linear_1_matmul_1_readvariableop_resource* 
_output_shapes
:
АА*
dtype02>
<LayerNormAndResidualMLP/mlp/linear_1/MatMul_1/ReadVariableOpУ
-LayerNormAndResidualMLP/mlp/linear_1/MatMul_1MatMul0LayerNormAndResidualMLP/mlp/Relu_1:activations:0DLayerNormAndResidualMLP/mlp/linear_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2/
-LayerNormAndResidualMLP/mlp/linear_1/MatMul_1Ў
9LayerNormAndResidualMLP/mlp/linear_1/Add_1/ReadVariableOpReadVariableOpBlayernormandresidualmlp_mlp_linear_1_add_1_readvariableop_resource*
_output_shapes	
:А*
dtype02;
9LayerNormAndResidualMLP/mlp/linear_1/Add_1/ReadVariableOpО
*LayerNormAndResidualMLP/mlp/linear_1/Add_1Add7LayerNormAndResidualMLP/mlp/linear_1/MatMul_1:product:0ALayerNormAndResidualMLP/mlp/linear_1/Add_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2,
*LayerNormAndResidualMLP/mlp/linear_1/Add_1н
6LayerNormAndResidualMLP/ResidualLayernormWrapper/add_1AddV2.LayerNormAndResidualMLP/mlp/linear_1/Add_1:z:0OLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/add_1:z:0*
T0*(
_output_shapes
:         А28
6LayerNormAndResidualMLP/ResidualLayernormWrapper/add_1П
\LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
         2^
\LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_1/mean/reduction_indicesЖ
JLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_1/meanMean:LayerNormAndResidualMLP/ResidualLayernormWrapper/add_1:z:0eLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_1/mean/reduction_indices:output:0*
T0*'
_output_shapes
:         *
	keep_dims(2L
JLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_1/mean┐
RLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_1/StopGradientStopGradientSLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_1/mean:output:0*
T0*'
_output_shapes
:         2T
RLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_1/StopGradientУ
WLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_1/SquaredDifferenceSquaredDifference:LayerNormAndResidualMLP/ResidualLayernormWrapper/add_1:z:0[LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_1/StopGradient:output:0*
T0*(
_output_shapes
:         А2Y
WLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_1/SquaredDifferenceЧ
`LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
         2b
`LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_1/variance/reduction_indices│
NLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_1/varianceMean[LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_1/SquaredDifference:z:0iLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_1/variance/reduction_indices:output:0*
T0*'
_output_shapes
:         *
	keep_dims(2P
NLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_1/varianceу
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *м┼'72O
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/add/yЖ
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/addAddV2WLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_1/variance:output:0VLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/add/y:output:0*
T0*'
_output_shapes
:         2M
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/addк
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/RsqrtRsqrtOLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/add:z:0*
T0*'
_output_shapes
:         2O
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/Rsqrt┘
ZLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/mul/ReadVariableOpReadVariableOpclayernormandresidualmlp_residuallayernormwrapper_layer_norm_batchnorm_1_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02\
ZLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/mul/ReadVariableOpЛ
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/mulMulQLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/Rsqrt:y:0bLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/mul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2M
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/mulх
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/mul_1Mul:LayerNormAndResidualMLP/ResidualLayernormWrapper/add_1:z:0OLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/mul:z:0*
T0*(
_output_shapes
:         А2O
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/mul_1■
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/mul_2MulSLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_1/mean:output:0OLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/mul:z:0*
T0*(
_output_shapes
:         А2O
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/mul_2ш
_LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/Identity/ReadVariableOpReadVariableOphlayernormandresidualmlp_residuallayernormwrapper_layer_norm_batchnorm_1_identity_readvariableop_resource*
_output_shapes	
:А*
dtype02a
_LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/Identity/ReadVariableOp┐
PLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/IdentityIdentitygLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/Identity/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2R
PLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/IdentityВ
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/subSubYLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/Identity:output:0QLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/mul_2:z:0*
T0*(
_output_shapes
:         А2M
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/sub■
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/add_1AddV2QLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/mul_1:z:0OLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/sub:z:0*
T0*(
_output_shapes
:         А2O
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/add_1Д
<LayerNormAndResidualMLP/mlp/linear_0/MatMul_3/ReadVariableOpReadVariableOpElayernormandresidualmlp_mlp_linear_0_matmul_3_readvariableop_resource* 
_output_shapes
:
АА*
dtype02>
<LayerNormAndResidualMLP/mlp/linear_0/MatMul_3/ReadVariableOp┤
-LayerNormAndResidualMLP/mlp/linear_0/MatMul_3MatMulQLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/add_1:z:0DLayerNormAndResidualMLP/mlp/linear_0/MatMul_3/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2/
-LayerNormAndResidualMLP/mlp/linear_0/MatMul_3Ў
9LayerNormAndResidualMLP/mlp/linear_0/Add_3/ReadVariableOpReadVariableOpBlayernormandresidualmlp_mlp_linear_0_add_3_readvariableop_resource*
_output_shapes	
:А*
dtype02;
9LayerNormAndResidualMLP/mlp/linear_0/Add_3/ReadVariableOpО
*LayerNormAndResidualMLP/mlp/linear_0/Add_3Add7LayerNormAndResidualMLP/mlp/linear_0/MatMul_3:product:0ALayerNormAndResidualMLP/mlp/linear_0/Add_3/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2,
*LayerNormAndResidualMLP/mlp/linear_0/Add_3│
"LayerNormAndResidualMLP/mlp/Relu_2Relu.LayerNormAndResidualMLP/mlp/linear_0/Add_3:z:0*
T0*(
_output_shapes
:         А2$
"LayerNormAndResidualMLP/mlp/Relu_2Д
<LayerNormAndResidualMLP/mlp/linear_1/MatMul_2/ReadVariableOpReadVariableOpElayernormandresidualmlp_mlp_linear_1_matmul_2_readvariableop_resource* 
_output_shapes
:
АА*
dtype02>
<LayerNormAndResidualMLP/mlp/linear_1/MatMul_2/ReadVariableOpУ
-LayerNormAndResidualMLP/mlp/linear_1/MatMul_2MatMul0LayerNormAndResidualMLP/mlp/Relu_2:activations:0DLayerNormAndResidualMLP/mlp/linear_1/MatMul_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2/
-LayerNormAndResidualMLP/mlp/linear_1/MatMul_2Ў
9LayerNormAndResidualMLP/mlp/linear_1/Add_2/ReadVariableOpReadVariableOpBlayernormandresidualmlp_mlp_linear_1_add_2_readvariableop_resource*
_output_shapes	
:А*
dtype02;
9LayerNormAndResidualMLP/mlp/linear_1/Add_2/ReadVariableOpО
*LayerNormAndResidualMLP/mlp/linear_1/Add_2Add7LayerNormAndResidualMLP/mlp/linear_1/MatMul_2:product:0ALayerNormAndResidualMLP/mlp/linear_1/Add_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2,
*LayerNormAndResidualMLP/mlp/linear_1/Add_2п
6LayerNormAndResidualMLP/ResidualLayernormWrapper/add_2AddV2.LayerNormAndResidualMLP/mlp/linear_1/Add_2:z:0QLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/add_1:z:0*
T0*(
_output_shapes
:         А28
6LayerNormAndResidualMLP/ResidualLayernormWrapper/add_2П
\LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_2/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
         2^
\LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_2/mean/reduction_indicesЖ
JLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_2/meanMean:LayerNormAndResidualMLP/ResidualLayernormWrapper/add_2:z:0eLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_2/mean/reduction_indices:output:0*
T0*'
_output_shapes
:         *
	keep_dims(2L
JLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_2/mean┐
RLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_2/StopGradientStopGradientSLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_2/mean:output:0*
T0*'
_output_shapes
:         2T
RLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_2/StopGradientУ
WLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_2/SquaredDifferenceSquaredDifference:LayerNormAndResidualMLP/ResidualLayernormWrapper/add_2:z:0[LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_2/StopGradient:output:0*
T0*(
_output_shapes
:         А2Y
WLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_2/SquaredDifferenceЧ
`LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_2/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
         2b
`LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_2/variance/reduction_indices│
NLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_2/varianceMean[LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_2/SquaredDifference:z:0iLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_2/variance/reduction_indices:output:0*
T0*'
_output_shapes
:         *
	keep_dims(2P
NLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_2/varianceу
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *м┼'72O
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/add/yЖ
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/addAddV2WLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_2/variance:output:0VLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/add/y:output:0*
T0*'
_output_shapes
:         2M
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/addк
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/RsqrtRsqrtOLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/add:z:0*
T0*'
_output_shapes
:         2O
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/Rsqrt┘
ZLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/mul/ReadVariableOpReadVariableOpclayernormandresidualmlp_residuallayernormwrapper_layer_norm_batchnorm_2_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02\
ZLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/mul/ReadVariableOpЛ
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/mulMulQLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/Rsqrt:y:0bLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/mul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2M
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/mulх
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/mul_1Mul:LayerNormAndResidualMLP/ResidualLayernormWrapper/add_2:z:0OLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/mul:z:0*
T0*(
_output_shapes
:         А2O
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/mul_1■
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/mul_2MulSLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_2/mean:output:0OLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/mul:z:0*
T0*(
_output_shapes
:         А2O
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/mul_2ш
_LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/Identity/ReadVariableOpReadVariableOphlayernormandresidualmlp_residuallayernormwrapper_layer_norm_batchnorm_2_identity_readvariableop_resource*
_output_shapes	
:А*
dtype02a
_LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/Identity/ReadVariableOp┐
PLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/IdentityIdentitygLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/Identity/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2R
PLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/IdentityВ
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/subSubYLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/Identity:output:0QLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/mul_2:z:0*
T0*(
_output_shapes
:         А2M
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/sub■
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/add_1AddV2QLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/mul_1:z:0OLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/sub:z:0*
T0*(
_output_shapes
:         А2O
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/add_1Д
<LayerNormAndResidualMLP/mlp/linear_0/MatMul_4/ReadVariableOpReadVariableOpElayernormandresidualmlp_mlp_linear_0_matmul_4_readvariableop_resource* 
_output_shapes
:
АА*
dtype02>
<LayerNormAndResidualMLP/mlp/linear_0/MatMul_4/ReadVariableOp┤
-LayerNormAndResidualMLP/mlp/linear_0/MatMul_4MatMulQLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/add_1:z:0DLayerNormAndResidualMLP/mlp/linear_0/MatMul_4/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2/
-LayerNormAndResidualMLP/mlp/linear_0/MatMul_4Ў
9LayerNormAndResidualMLP/mlp/linear_0/Add_4/ReadVariableOpReadVariableOpBlayernormandresidualmlp_mlp_linear_0_add_4_readvariableop_resource*
_output_shapes	
:А*
dtype02;
9LayerNormAndResidualMLP/mlp/linear_0/Add_4/ReadVariableOpО
*LayerNormAndResidualMLP/mlp/linear_0/Add_4Add7LayerNormAndResidualMLP/mlp/linear_0/MatMul_4:product:0ALayerNormAndResidualMLP/mlp/linear_0/Add_4/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2,
*LayerNormAndResidualMLP/mlp/linear_0/Add_4│
"LayerNormAndResidualMLP/mlp/Relu_3Relu.LayerNormAndResidualMLP/mlp/linear_0/Add_4:z:0*
T0*(
_output_shapes
:         А2$
"LayerNormAndResidualMLP/mlp/Relu_3Д
<LayerNormAndResidualMLP/mlp/linear_1/MatMul_3/ReadVariableOpReadVariableOpElayernormandresidualmlp_mlp_linear_1_matmul_3_readvariableop_resource* 
_output_shapes
:
АА*
dtype02>
<LayerNormAndResidualMLP/mlp/linear_1/MatMul_3/ReadVariableOpУ
-LayerNormAndResidualMLP/mlp/linear_1/MatMul_3MatMul0LayerNormAndResidualMLP/mlp/Relu_3:activations:0DLayerNormAndResidualMLP/mlp/linear_1/MatMul_3/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2/
-LayerNormAndResidualMLP/mlp/linear_1/MatMul_3Ў
9LayerNormAndResidualMLP/mlp/linear_1/Add_3/ReadVariableOpReadVariableOpBlayernormandresidualmlp_mlp_linear_1_add_3_readvariableop_resource*
_output_shapes	
:А*
dtype02;
9LayerNormAndResidualMLP/mlp/linear_1/Add_3/ReadVariableOpО
*LayerNormAndResidualMLP/mlp/linear_1/Add_3Add7LayerNormAndResidualMLP/mlp/linear_1/MatMul_3:product:0ALayerNormAndResidualMLP/mlp/linear_1/Add_3/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2,
*LayerNormAndResidualMLP/mlp/linear_1/Add_3п
6LayerNormAndResidualMLP/ResidualLayernormWrapper/add_3AddV2.LayerNormAndResidualMLP/mlp/linear_1/Add_3:z:0QLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/add_1:z:0*
T0*(
_output_shapes
:         А28
6LayerNormAndResidualMLP/ResidualLayernormWrapper/add_3П
\LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_3/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
         2^
\LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_3/mean/reduction_indicesЖ
JLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_3/meanMean:LayerNormAndResidualMLP/ResidualLayernormWrapper/add_3:z:0eLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_3/mean/reduction_indices:output:0*
T0*'
_output_shapes
:         *
	keep_dims(2L
JLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_3/mean┐
RLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_3/StopGradientStopGradientSLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_3/mean:output:0*
T0*'
_output_shapes
:         2T
RLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_3/StopGradientУ
WLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_3/SquaredDifferenceSquaredDifference:LayerNormAndResidualMLP/ResidualLayernormWrapper/add_3:z:0[LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_3/StopGradient:output:0*
T0*(
_output_shapes
:         А2Y
WLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_3/SquaredDifferenceЧ
`LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_3/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
         2b
`LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_3/variance/reduction_indices│
NLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_3/varianceMean[LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_3/SquaredDifference:z:0iLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_3/variance/reduction_indices:output:0*
T0*'
_output_shapes
:         *
	keep_dims(2P
NLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_3/varianceу
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *м┼'72O
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/add/yЖ
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/addAddV2WLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_3/variance:output:0VLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/add/y:output:0*
T0*'
_output_shapes
:         2M
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/addк
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/RsqrtRsqrtOLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/add:z:0*
T0*'
_output_shapes
:         2O
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/Rsqrt┘
ZLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/mul/ReadVariableOpReadVariableOpclayernormandresidualmlp_residuallayernormwrapper_layer_norm_batchnorm_3_mul_readvariableop_resource*
_output_shapes	
:А*
dtype02\
ZLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/mul/ReadVariableOpЛ
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/mulMulQLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/Rsqrt:y:0bLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/mul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2M
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/mulх
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/mul_1Mul:LayerNormAndResidualMLP/ResidualLayernormWrapper/add_3:z:0OLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/mul:z:0*
T0*(
_output_shapes
:         А2O
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/mul_1■
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/mul_2MulSLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_3/mean:output:0OLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/mul:z:0*
T0*(
_output_shapes
:         А2O
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/mul_2ш
_LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/Identity/ReadVariableOpReadVariableOphlayernormandresidualmlp_residuallayernormwrapper_layer_norm_batchnorm_3_identity_readvariableop_resource*
_output_shapes	
:А*
dtype02a
_LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/Identity/ReadVariableOp┐
PLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/IdentityIdentitygLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/Identity/ReadVariableOp:value:0*
T0*
_output_shapes	
:А2R
PLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/IdentityВ
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/subSubYLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/Identity:output:0QLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/mul_2:z:0*
T0*(
_output_shapes
:         А2M
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/sub■
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/add_1AddV2QLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/mul_1:z:0OLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/sub:z:0*
T0*(
_output_shapes
:         А2O
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/add_1Й
 MultivariateNormalDiagHead/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 MultivariateNormalDiagHead/ConstЇ
7MultivariateNormalDiagHead/linear/MatMul/ReadVariableOpReadVariableOp@multivariatenormaldiaghead_linear_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype029
7MultivariateNormalDiagHead/linear/MatMul/ReadVariableOpд
(MultivariateNormalDiagHead/linear/MatMulMatMulQLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/add_1:z:0?MultivariateNormalDiagHead/linear/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2*
(MultivariateNormalDiagHead/linear/MatMulц
4MultivariateNormalDiagHead/linear/Add/ReadVariableOpReadVariableOp=multivariatenormaldiaghead_linear_add_readvariableop_resource*
_output_shapes
:*
dtype026
4MultivariateNormalDiagHead/linear/Add/ReadVariableOp∙
%MultivariateNormalDiagHead/linear/AddAdd2MultivariateNormalDiagHead/linear/MatMul:product:0<MultivariateNormalDiagHead/linear/Add/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2'
%MultivariateNormalDiagHead/linear/Add·
9MultivariateNormalDiagHead/linear/MatMul_1/ReadVariableOpReadVariableOpBmultivariatenormaldiaghead_linear_matmul_1_readvariableop_resource*
_output_shapes
:	А*
dtype02;
9MultivariateNormalDiagHead/linear/MatMul_1/ReadVariableOpк
*MultivariateNormalDiagHead/linear/MatMul_1MatMulQLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/add_1:z:0AMultivariateNormalDiagHead/linear/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2,
*MultivariateNormalDiagHead/linear/MatMul_1ь
6MultivariateNormalDiagHead/linear/Add_1/ReadVariableOpReadVariableOp?multivariatenormaldiaghead_linear_add_1_readvariableop_resource*
_output_shapes
:*
dtype028
6MultivariateNormalDiagHead/linear/Add_1/ReadVariableOpБ
'MultivariateNormalDiagHead/linear/Add_1Add4MultivariateNormalDiagHead/linear/MatMul_1:product:0>MultivariateNormalDiagHead/linear/Add_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2)
'MultivariateNormalDiagHead/linear/Add_1╡
#MultivariateNormalDiagHead/SoftplusSoftplus+MultivariateNormalDiagHead/linear/Add_1:z:0*
T0*'
_output_shapes
:         2%
#MultivariateNormalDiagHead/Softplusж
%MultivariateNormalDiagHead/Softplus_1Softplus)MultivariateNormalDiagHead/Const:output:0*
T0*
_output_shapes
: 2'
%MultivariateNormalDiagHead/Softplus_1С
$MultivariateNormalDiagHead/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЪЩЩ>2&
$MultivariateNormalDiagHead/truediv/x╪
"MultivariateNormalDiagHead/truedivRealDiv-MultivariateNormalDiagHead/truediv/x:output:03MultivariateNormalDiagHead/Softplus_1:activations:0*
T0*
_output_shapes
: 2$
"MultivariateNormalDiagHead/truediv╘
MultivariateNormalDiagHead/mulMul1MultivariateNormalDiagHead/Softplus:activations:0&MultivariateNormalDiagHead/truediv:z:0*
T0*'
_output_shapes
:         2 
MultivariateNormalDiagHead/mulЙ
 MultivariateNormalDiagHead/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *╜7Ж52"
 MultivariateNormalDiagHead/add/y╩
MultivariateNormalDiagHead/addAddV2"MultivariateNormalDiagHead/mul:z:0)MultivariateNormalDiagHead/add/y:output:0*
T0*'
_output_shapes
:         2 
MultivariateNormalDiagHead/add╝
{MultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/range_dimension_tensor/ConstConst*
_output_shapes
: *
dtype0*
value	B :2}
{MultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/range_dimension_tensor/Const№
WMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/event_shapeConst*
_output_shapes
:*
dtype0*
valueB:2Y
WMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/event_shapeЗ
ЧMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/ShapeShape"MultivariateNormalDiagHead/add:z:0*
T0*
_output_shapes
:2Ъ
ЧMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/Shapeд
еMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2и
еMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stackЯ
зMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2к
зMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_1Я
зMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2к
зMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_2╒
ЯMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_sliceStridedSliceаMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/Shape:output:0оMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack:output:0░MultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_1:output:0░MultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2в
ЯMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_sliceк
бMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/values_1PackиMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice:output:0*
N*
T0*
_output_shapes
:2д
бMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/values_1Г
ЭMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2а
ЭMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/axisъ
ШMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concatConcatV2аMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/Shape:output:0кMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/values_1:output:0жMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/axis:output:0*
N*
T0*
_output_shapes
:2Ы
ШMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat█
ЕMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2И
ЕMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stackш
ЗMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
■        2К
ЗMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_1▀
ЗMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2К
ЗMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_2▓
MultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_sliceStridedSliceбMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat:output:0ОMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack:output:0РMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_1:output:0РMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2Б
MultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice 
QMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/ShapeShape)MultivariateNormalDiagHead/linear/Add:z:0*
T0*
_output_shapes
:2S
QMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/ShapeМ
_MultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2a
_MultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stackЩ
aMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
         2c
aMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_1Р
aMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2c
aMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_2и
YMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_sliceStridedSliceZMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/Shape:output:0hMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack:output:0jMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_1:output:0jMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2[
YMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_sliceО
wMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/prefer_static_broadcast_shape/BroadcastArgsBroadcastArgsИMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice:output:0bMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice:output:0*
_output_shapes
:2y
wMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/prefer_static_broadcast_shape/BroadcastArgs├
=MultivariateNormalDiagHead/MultivariateNormalDiag/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2?
=MultivariateNormalDiagHead/MultivariateNormalDiag/zeros/Constю
7MultivariateNormalDiagHead/MultivariateNormalDiag/zerosFill|MultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/prefer_static_broadcast_shape/BroadcastArgs:r0:0FMultivariateNormalDiagHead/MultivariateNormalDiag/zeros/Const:output:0*
T0*#
_output_shapes
:         29
7MultivariateNormalDiagHead/MultivariateNormalDiag/zeros╡
6MultivariateNormalDiagHead/MultivariateNormalDiag/onesConst*
_output_shapes
: *
dtype0*
valueB
 *  А?28
6MultivariateNormalDiagHead/MultivariateNormalDiag/ones▓
6MultivariateNormalDiagHead/MultivariateNormalDiag/zeroConst*
_output_shapes
: *
dtype0*
value	B : 28
6MultivariateNormalDiagHead/MultivariateNormalDiag/zero╡
7MultivariateNormalDiagHead/MultivariateNormalDiag/emptyConst*
_output_shapes
: *
dtype0*
valueB 29
7MultivariateNormalDiagHead/MultivariateNormalDiag/emptyГ
^stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB 2`
^stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/sample_shapeН
тstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal/is_scalar_batch/is_scalar_batchConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2х
тstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal/is_scalar_batch/is_scalar_batchК
bstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/pick_vector/condConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2d
bstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/pick_vector/condв
jstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/pick_vector/false_vectorConst*
_output_shapes
:*
dtype0*
valueB:2l
jstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/pick_vector/false_vectorО
dstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/pick_vector_1/condConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2f
dstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/pick_vector_1/condд
kstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/pick_vector_1/true_vectorConst*
_output_shapes
:*
dtype0*
valueB:2m
kstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/pick_vector_1/true_vectorЕ
┌stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/sample_shape/xConst*
_output_shapes
:*
dtype0*
valueB:2▌
┌stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/sample_shape/xМ
Ъstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/sample_shape/xConst*
_output_shapes
:*
dtype0*
valueB"      2Э
Ъstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/sample_shape/xЩ
Сstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/ShapeShape@MultivariateNormalDiagHead/MultivariateNormalDiag/zeros:output:0*
T0*
_output_shapes
:2Ф
Сstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/ShapeВ
Ьstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/BroadcastArgs/s1Const*
_output_shapes
: *
dtype0*
valueB 2Я
Ьstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/BroadcastArgs/s1л	
Щstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/BroadcastArgsBroadcastArgsЪstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/Shape:output:0еstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/BroadcastArgs/s1:output:0*
_output_shapes
:2Ь
Щstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/BroadcastArgsЗ
Ыstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:2Ю
Ыstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/concat/values_0ў
Чstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2Ъ
Чstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/concat/axis╨
Тstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/concatConcatV2дstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/concat/values_0:output:0Юstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/BroadcastArgs:r0:0аstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2Х
Тstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/concatЦ
еstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2и
еstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/normal/random_normal/meanЪ
зstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2к
зstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/normal/random_normal/stddevц
╡stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormalЫstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/concat:output:0*
T0*'
_output_shapes
:         *
dtype02╕
╡stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/normal/random_normal/RandomStandardNormal№	
дstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/normal/random_normal/mulMul╛stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/normal/random_normal/RandomStandardNormal:output:0░stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/normal/random_normal/stddev:output:0*
T0*'
_output_shapes
:         2з
дstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/normal/random_normal/mul▄	
аstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/normal/random_normalAddиstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/normal/random_normal/mul:z:0оstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/normal/random_normal/mean:output:0*
T0*'
_output_shapes
:         2г
аstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/normal/random_normal╞
Пstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/mulMulдstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/normal/random_normal:z:0?MultivariateNormalDiagHead/MultivariateNormalDiag/ones:output:0*
T0*'
_output_shapes
:         2Т
Пstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/mul╕
Пstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/addAddV2Уstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/mul:z:0@MultivariateNormalDiagHead/MultivariateNormalDiag/zeros:output:0*
T0*'
_output_shapes
:         2Т
Пstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/addё
Уstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/Shape_1ShapeУstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/add:z:0*
T0*
_output_shapes
:2Ц
Уstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/Shape_1П
Яstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2в
Яstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/strided_slice/stackУ
бstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2д
бstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/strided_slice/stack_1У
бstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2д
бstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/strided_slice/stack_2п
Щstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/strided_sliceStridedSliceЬstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/Shape_1:output:0иstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/strided_slice/stack:output:0кstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/strided_slice/stack_1:output:0кstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2Ь
Щstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/strided_slice√
Щstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2Ь
Щstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/concat_1/axis┘
Фstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/concat_1ConcatV2гstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/sample_shape/x:output:0вstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/strided_slice:output:0вstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2Ч
Фstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/concat_1д	
Уstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/ReshapeReshapeУstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/add:z:0Эstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/concat_1:output:0*
T0*+
_output_shapes
:         2Ц
Уstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/ReshapeР
┌stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2▌
┌stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/transpose/perm∙
╒stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/transpose	TransposeЬstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/Reshape:output:0уstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/transpose/perm:output:0*
T0*+
_output_shapes
:         2╪
╒stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/transpose│
╤stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/ShapeShape┘stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/transpose:y:0*
T0*
_output_shapes
:2╘
╤stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/ShapeП
▀stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2т
▀stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/strided_slice/stackУ
сstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2ф
сstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/strided_slice/stack_1У
сstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2ф
сstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/strided_slice/stack_2н
┘stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/strided_sliceStridedSlice┌stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/Shape:output:0шstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/strided_slice/stack:output:0ъstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/strided_slice/stack_1:output:0ъstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2▄
┘stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/strided_sliceў
╫stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2┌
╫stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/concat/axisУ	
╥stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/concatConcatV2уstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/sample_shape/x:output:0тstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/strided_slice:output:0рstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2╒
╥stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/concatи
╙stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/ReshapeReshape┘stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/transpose:y:0█stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/concat:output:0*
T0*+
_output_shapes
:         2╓
╙stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/Reshape┐
Wstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/ShapeShape▄stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/Reshape:output:0*
T0*
_output_shapes
:2Y
Wstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/ShapeШ
estochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2g
estochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/strided_slice/stackЬ
gstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2i
gstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/strided_slice/stack_1Ь
gstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2i
gstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/strided_slice/stack_2╩
_stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/strided_sliceStridedSlice`stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/Shape:output:0nstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/strided_slice/stack:output:0pstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/strided_slice/stack_1:output:0pstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2a
_stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/strided_sliceА
]stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2_
]stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/concat/axisй
Xstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/concatConcatV2gstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/sample_shape:output:0hstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/strided_slice:output:0fstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2Z
Xstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/concat╡
Ystochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/ReshapeReshape▄stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/Reshape:output:0astochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/concat:output:0*
T0*'
_output_shapes
:         2[
Ystochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/Reshapeю
Уstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_chain_of_MultivariateNormalDiagHead_MultivariateNormalDiag_shift_of_MultivariateNormalDiagHead_MultivariateNormalDiag_scale_matvec_linear_operator/forward/MultivariateNormalDiagHead_MultivariateNormalDiag_scale_matvec_linear_operator/forward/LinearOperatorDiag/matvec/mulMul"MultivariateNormalDiagHead/add:z:0bstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/Reshape:output:0*
T0*'
_output_shapes
:         2Ц
Уstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_chain_of_MultivariateNormalDiagHead_MultivariateNormalDiag_shift_of_MultivariateNormalDiagHead_MultivariateNormalDiag_scale_matvec_linear_operator/forward/MultivariateNormalDiagHead_MultivariateNormalDiag_scale_matvec_linear_operator/forward/LinearOperatorDiag/matvec/mul╦	
тstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_chain_of_MultivariateNormalDiagHead_MultivariateNormalDiag_shift_of_MultivariateNormalDiagHead_MultivariateNormalDiag_scale_matvec_linear_operator/forward/MultivariateNormalDiagHead_MultivariateNormalDiag_shift/forward/addAddV2Чstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_chain_of_MultivariateNormalDiagHead_MultivariateNormalDiag_shift_of_MultivariateNormalDiagHead_MultivariateNormalDiag_scale_matvec_linear_operator/forward/MultivariateNormalDiagHead_MultivariateNormalDiag_scale_matvec_linear_operator/forward/LinearOperatorDiag/matvec/mul:z:0)MultivariateNormalDiagHead/linear/Add:z:0*
T0*'
_output_shapes
:         2х
тstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_chain_of_MultivariateNormalDiagHead_MultivariateNormalDiag_shift_of_MultivariateNormalDiagHead_MultivariateNormalDiag_scale_matvec_linear_operator/forward/MultivariateNormalDiagHead_MultivariateNormalDiag_shift/forward/add╗
IdentityIdentityцstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_chain_of_MultivariateNormalDiagHead_MultivariateNormalDiag_shift_of_MultivariateNormalDiagHead_MultivariateNormalDiag_scale_matvec_linear_operator/forward/MultivariateNormalDiagHead_MultivariateNormalDiag_shift/forward/add:z:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*щ
_input_shapes╫
╘:         :         :         :         :::::::::::::::::::::::::::::::::::\ X
'
_output_shapes
:         
-
_user_specified_nameargs_0/joint_angles:VR
'
_output_shapes
:         
'
_user_specified_nameargs_0/target:WS
'
_output_shapes
:         
(
_user_specified_nameargs_0/upright:XT
'
_output_shapes
:         
)
_user_specified_nameargs_0/velocity:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	
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
: :
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: 
┌Ь
У
#__inference__traced_restore_2551977
file_prefix0
,assignvariableop_control_network_normalize_b2
.assignvariableop_1_control_network_normalize_w8
4assignvariableop_2_control_network_layer_norm_offset7
3assignvariableop_3_control_network_layer_norm_scale=
9assignvariableop_4_layernormandresidualmlp_mlp_linear_0_b=
9assignvariableop_5_layernormandresidualmlp_mlp_linear_0_w?
;assignvariableop_6_layernormandresidualmlp_mlp_linear_0_b_1?
;assignvariableop_7_layernormandresidualmlp_mlp_linear_0_w_1=
9assignvariableop_8_layernormandresidualmlp_mlp_linear_1_b=
9assignvariableop_9_layernormandresidualmlp_mlp_linear_1_wZ
Vassignvariableop_10_layernormandresidualmlp_residuallayernormwrapper_layer_norm_offsetY
Uassignvariableop_11_layernormandresidualmlp_residuallayernormwrapper_layer_norm_scale@
<assignvariableop_12_layernormandresidualmlp_mlp_linear_0_b_2@
<assignvariableop_13_layernormandresidualmlp_mlp_linear_0_w_2@
<assignvariableop_14_layernormandresidualmlp_mlp_linear_1_b_1@
<assignvariableop_15_layernormandresidualmlp_mlp_linear_1_w_1\
Xassignvariableop_16_layernormandresidualmlp_residuallayernormwrapper_layer_norm_offset_1[
Wassignvariableop_17_layernormandresidualmlp_residuallayernormwrapper_layer_norm_scale_1@
<assignvariableop_18_layernormandresidualmlp_mlp_linear_0_b_3@
<assignvariableop_19_layernormandresidualmlp_mlp_linear_0_w_3@
<assignvariableop_20_layernormandresidualmlp_mlp_linear_1_b_2@
<assignvariableop_21_layernormandresidualmlp_mlp_linear_1_w_2\
Xassignvariableop_22_layernormandresidualmlp_residuallayernormwrapper_layer_norm_offset_2[
Wassignvariableop_23_layernormandresidualmlp_residuallayernormwrapper_layer_norm_scale_2@
<assignvariableop_24_layernormandresidualmlp_mlp_linear_0_b_4@
<assignvariableop_25_layernormandresidualmlp_mlp_linear_0_w_4@
<assignvariableop_26_layernormandresidualmlp_mlp_linear_1_b_3@
<assignvariableop_27_layernormandresidualmlp_mlp_linear_1_w_3\
Xassignvariableop_28_layernormandresidualmlp_residuallayernormwrapper_layer_norm_offset_3[
Wassignvariableop_29_layernormandresidualmlp_residuallayernormwrapper_layer_norm_scale_3;
7assignvariableop_30_multivariatenormaldiaghead_linear_b;
7assignvariableop_31_multivariatenormaldiaghead_linear_w=
9assignvariableop_32_multivariatenormaldiaghead_linear_b_1=
9assignvariableop_33_multivariatenormaldiaghead_linear_w_1
identity_35ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_23вAssignVariableOp_24вAssignVariableOp_25вAssignVariableOp_26вAssignVariableOp_27вAssignVariableOp_28вAssignVariableOp_29вAssignVariableOp_3вAssignVariableOp_30вAssignVariableOp_31вAssignVariableOp_32вAssignVariableOp_33вAssignVariableOp_4вAssignVariableOp_5вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9в	RestoreV2вRestoreV2_1У
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*Я
valueХBТ"B'_variables/0/.ATTRIBUTES/VARIABLE_VALUEB'_variables/1/.ATTRIBUTES/VARIABLE_VALUEB'_variables/2/.ATTRIBUTES/VARIABLE_VALUEB'_variables/3/.ATTRIBUTES/VARIABLE_VALUEB'_variables/4/.ATTRIBUTES/VARIABLE_VALUEB'_variables/5/.ATTRIBUTES/VARIABLE_VALUEB'_variables/6/.ATTRIBUTES/VARIABLE_VALUEB'_variables/7/.ATTRIBUTES/VARIABLE_VALUEB'_variables/8/.ATTRIBUTES/VARIABLE_VALUEB'_variables/9/.ATTRIBUTES/VARIABLE_VALUEB(_variables/10/.ATTRIBUTES/VARIABLE_VALUEB(_variables/11/.ATTRIBUTES/VARIABLE_VALUEB(_variables/12/.ATTRIBUTES/VARIABLE_VALUEB(_variables/13/.ATTRIBUTES/VARIABLE_VALUEB(_variables/14/.ATTRIBUTES/VARIABLE_VALUEB(_variables/15/.ATTRIBUTES/VARIABLE_VALUEB(_variables/16/.ATTRIBUTES/VARIABLE_VALUEB(_variables/17/.ATTRIBUTES/VARIABLE_VALUEB(_variables/18/.ATTRIBUTES/VARIABLE_VALUEB(_variables/19/.ATTRIBUTES/VARIABLE_VALUEB(_variables/20/.ATTRIBUTES/VARIABLE_VALUEB(_variables/21/.ATTRIBUTES/VARIABLE_VALUEB(_variables/22/.ATTRIBUTES/VARIABLE_VALUEB(_variables/23/.ATTRIBUTES/VARIABLE_VALUEB(_variables/24/.ATTRIBUTES/VARIABLE_VALUEB(_variables/25/.ATTRIBUTES/VARIABLE_VALUEB(_variables/26/.ATTRIBUTES/VARIABLE_VALUEB(_variables/27/.ATTRIBUTES/VARIABLE_VALUEB(_variables/28/.ATTRIBUTES/VARIABLE_VALUEB(_variables/29/.ATTRIBUTES/VARIABLE_VALUEB(_variables/30/.ATTRIBUTES/VARIABLE_VALUEB(_variables/31/.ATTRIBUTES/VARIABLE_VALUEB(_variables/32/.ATTRIBUTES/VARIABLE_VALUEB(_variables/33/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names╥
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices╪
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Ю
_output_shapesЛ
И::::::::::::::::::::::::::::::::::*0
dtypes&
$2"2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

IdentityЬ
AssignVariableOpAssignVariableOp,assignvariableop_control_network_normalize_bIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1д
AssignVariableOp_1AssignVariableOp.assignvariableop_1_control_network_normalize_wIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2к
AssignVariableOp_2AssignVariableOp4assignvariableop_2_control_network_layer_norm_offsetIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3й
AssignVariableOp_3AssignVariableOp3assignvariableop_3_control_network_layer_norm_scaleIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4п
AssignVariableOp_4AssignVariableOp9assignvariableop_4_layernormandresidualmlp_mlp_linear_0_bIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5п
AssignVariableOp_5AssignVariableOp9assignvariableop_5_layernormandresidualmlp_mlp_linear_0_wIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6▒
AssignVariableOp_6AssignVariableOp;assignvariableop_6_layernormandresidualmlp_mlp_linear_0_b_1Identity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7▒
AssignVariableOp_7AssignVariableOp;assignvariableop_7_layernormandresidualmlp_mlp_linear_0_w_1Identity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8п
AssignVariableOp_8AssignVariableOp9assignvariableop_8_layernormandresidualmlp_mlp_linear_1_bIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9п
AssignVariableOp_9AssignVariableOp9assignvariableop_9_layernormandresidualmlp_mlp_linear_1_wIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10╧
AssignVariableOp_10AssignVariableOpVassignvariableop_10_layernormandresidualmlp_residuallayernormwrapper_layer_norm_offsetIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11╬
AssignVariableOp_11AssignVariableOpUassignvariableop_11_layernormandresidualmlp_residuallayernormwrapper_layer_norm_scaleIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12╡
AssignVariableOp_12AssignVariableOp<assignvariableop_12_layernormandresidualmlp_mlp_linear_0_b_2Identity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13╡
AssignVariableOp_13AssignVariableOp<assignvariableop_13_layernormandresidualmlp_mlp_linear_0_w_2Identity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14╡
AssignVariableOp_14AssignVariableOp<assignvariableop_14_layernormandresidualmlp_mlp_linear_1_b_1Identity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15╡
AssignVariableOp_15AssignVariableOp<assignvariableop_15_layernormandresidualmlp_mlp_linear_1_w_1Identity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16╤
AssignVariableOp_16AssignVariableOpXassignvariableop_16_layernormandresidualmlp_residuallayernormwrapper_layer_norm_offset_1Identity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17╨
AssignVariableOp_17AssignVariableOpWassignvariableop_17_layernormandresidualmlp_residuallayernormwrapper_layer_norm_scale_1Identity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18╡
AssignVariableOp_18AssignVariableOp<assignvariableop_18_layernormandresidualmlp_mlp_linear_0_b_3Identity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19╡
AssignVariableOp_19AssignVariableOp<assignvariableop_19_layernormandresidualmlp_mlp_linear_0_w_3Identity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:2
Identity_20╡
AssignVariableOp_20AssignVariableOp<assignvariableop_20_layernormandresidualmlp_mlp_linear_1_b_2Identity_20:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21╡
AssignVariableOp_21AssignVariableOp<assignvariableop_21_layernormandresidualmlp_mlp_linear_1_w_2Identity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21_
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:2
Identity_22╤
AssignVariableOp_22AssignVariableOpXassignvariableop_22_layernormandresidualmlp_residuallayernormwrapper_layer_norm_offset_2Identity_22:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_22_
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:2
Identity_23╨
AssignVariableOp_23AssignVariableOpWassignvariableop_23_layernormandresidualmlp_residuallayernormwrapper_layer_norm_scale_2Identity_23:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_23_
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:2
Identity_24╡
AssignVariableOp_24AssignVariableOp<assignvariableop_24_layernormandresidualmlp_mlp_linear_0_b_4Identity_24:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_24_
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:2
Identity_25╡
AssignVariableOp_25AssignVariableOp<assignvariableop_25_layernormandresidualmlp_mlp_linear_0_w_4Identity_25:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_25_
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:2
Identity_26╡
AssignVariableOp_26AssignVariableOp<assignvariableop_26_layernormandresidualmlp_mlp_linear_1_b_3Identity_26:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_26_
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:2
Identity_27╡
AssignVariableOp_27AssignVariableOp<assignvariableop_27_layernormandresidualmlp_mlp_linear_1_w_3Identity_27:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_27_
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:2
Identity_28╤
AssignVariableOp_28AssignVariableOpXassignvariableop_28_layernormandresidualmlp_residuallayernormwrapper_layer_norm_offset_3Identity_28:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_28_
Identity_29IdentityRestoreV2:tensors:29*
T0*
_output_shapes
:2
Identity_29╨
AssignVariableOp_29AssignVariableOpWassignvariableop_29_layernormandresidualmlp_residuallayernormwrapper_layer_norm_scale_3Identity_29:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_29_
Identity_30IdentityRestoreV2:tensors:30*
T0*
_output_shapes
:2
Identity_30░
AssignVariableOp_30AssignVariableOp7assignvariableop_30_multivariatenormaldiaghead_linear_bIdentity_30:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_30_
Identity_31IdentityRestoreV2:tensors:31*
T0*
_output_shapes
:2
Identity_31░
AssignVariableOp_31AssignVariableOp7assignvariableop_31_multivariatenormaldiaghead_linear_wIdentity_31:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_31_
Identity_32IdentityRestoreV2:tensors:32*
T0*
_output_shapes
:2
Identity_32▓
AssignVariableOp_32AssignVariableOp9assignvariableop_32_multivariatenormaldiaghead_linear_b_1Identity_32:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_32_
Identity_33IdentityRestoreV2:tensors:33*
T0*
_output_shapes
:2
Identity_33▓
AssignVariableOp_33AssignVariableOp9assignvariableop_33_multivariatenormaldiaghead_linear_w_1Identity_33:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_33и
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_namesФ
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slices─
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp╩
Identity_34Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_34╫
Identity_35IdentityIdentity_34:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_35"#
identity_35Identity_35:output:0*Я
_input_shapesН
К: ::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_33AssignVariableOp_332(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	
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
: :
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: 
╞
п
__inference___call___82212
args_0_joint_angles
args_0_target
args_0_upright
args_0_velocity
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
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32
identityИвStatefulPartitionedCall▄
StatefulPartitionedCallStatefulPartitionedCallargs_0_joint_anglesargs_0_targetargs_0_uprightargs_0_velocityunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32*1
Tin*
(2&*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *D
_read_only_resource_inputs&
$"	
 !"#$%*C
config_proto31

CPU

GPU 

TPU


TPU_SYSTEM2J 8*(
f#R!
__inference_wrapped_module_88362
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*щ
_input_shapes╫
╘:         :         :         :         ::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
'
_output_shapes
:         
-
_user_specified_nameargs_0/joint_angles:VR
'
_output_shapes
:         
'
_user_specified_nameargs_0/target:WS
'
_output_shapes
:         
(
_user_specified_nameargs_0/upright:XT
'
_output_shapes
:         
)
_user_specified_nameargs_0/velocity:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	
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
: :
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: 
Ж
-
__inference_initial_state_82520

args_0*
_input_shapes
: :> :

_output_shapes
: 
 
_user_specified_nameargs_0
РZ
╠
 __inference__traced_save_2551863
file_prefix:
6savev2_control_network_normalize_b_read_readvariableop:
6savev2_control_network_normalize_w_read_readvariableop@
<savev2_control_network_layer_norm_offset_read_readvariableop?
;savev2_control_network_layer_norm_scale_read_readvariableopE
Asavev2_layernormandresidualmlp_mlp_linear_0_b_read_readvariableopE
Asavev2_layernormandresidualmlp_mlp_linear_0_w_read_readvariableopG
Csavev2_layernormandresidualmlp_mlp_linear_0_b_1_read_readvariableopG
Csavev2_layernormandresidualmlp_mlp_linear_0_w_1_read_readvariableopE
Asavev2_layernormandresidualmlp_mlp_linear_1_b_read_readvariableopE
Asavev2_layernormandresidualmlp_mlp_linear_1_w_read_readvariableopa
]savev2_layernormandresidualmlp_residuallayernormwrapper_layer_norm_offset_read_readvariableop`
\savev2_layernormandresidualmlp_residuallayernormwrapper_layer_norm_scale_read_readvariableopG
Csavev2_layernormandresidualmlp_mlp_linear_0_b_2_read_readvariableopG
Csavev2_layernormandresidualmlp_mlp_linear_0_w_2_read_readvariableopG
Csavev2_layernormandresidualmlp_mlp_linear_1_b_1_read_readvariableopG
Csavev2_layernormandresidualmlp_mlp_linear_1_w_1_read_readvariableopc
_savev2_layernormandresidualmlp_residuallayernormwrapper_layer_norm_offset_1_read_readvariableopb
^savev2_layernormandresidualmlp_residuallayernormwrapper_layer_norm_scale_1_read_readvariableopG
Csavev2_layernormandresidualmlp_mlp_linear_0_b_3_read_readvariableopG
Csavev2_layernormandresidualmlp_mlp_linear_0_w_3_read_readvariableopG
Csavev2_layernormandresidualmlp_mlp_linear_1_b_2_read_readvariableopG
Csavev2_layernormandresidualmlp_mlp_linear_1_w_2_read_readvariableopc
_savev2_layernormandresidualmlp_residuallayernormwrapper_layer_norm_offset_2_read_readvariableopb
^savev2_layernormandresidualmlp_residuallayernormwrapper_layer_norm_scale_2_read_readvariableopG
Csavev2_layernormandresidualmlp_mlp_linear_0_b_4_read_readvariableopG
Csavev2_layernormandresidualmlp_mlp_linear_0_w_4_read_readvariableopG
Csavev2_layernormandresidualmlp_mlp_linear_1_b_3_read_readvariableopG
Csavev2_layernormandresidualmlp_mlp_linear_1_w_3_read_readvariableopc
_savev2_layernormandresidualmlp_residuallayernormwrapper_layer_norm_offset_3_read_readvariableopb
^savev2_layernormandresidualmlp_residuallayernormwrapper_layer_norm_scale_3_read_readvariableopB
>savev2_multivariatenormaldiaghead_linear_b_read_readvariableopB
>savev2_multivariatenormaldiaghead_linear_w_read_readvariableopD
@savev2_multivariatenormaldiaghead_linear_b_1_read_readvariableopD
@savev2_multivariatenormaldiaghead_linear_w_1_read_readvariableop
savev2_1_const

identity_1ИвMergeV2CheckpointsвSaveV2вSaveV2_1П
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
ConstН
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_fe865b6e204a42379a6fb6d0eea3241b/part2	
Const_1Л
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
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardж
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameН
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*Я
valueХBТ"B'_variables/0/.ATTRIBUTES/VARIABLE_VALUEB'_variables/1/.ATTRIBUTES/VARIABLE_VALUEB'_variables/2/.ATTRIBUTES/VARIABLE_VALUEB'_variables/3/.ATTRIBUTES/VARIABLE_VALUEB'_variables/4/.ATTRIBUTES/VARIABLE_VALUEB'_variables/5/.ATTRIBUTES/VARIABLE_VALUEB'_variables/6/.ATTRIBUTES/VARIABLE_VALUEB'_variables/7/.ATTRIBUTES/VARIABLE_VALUEB'_variables/8/.ATTRIBUTES/VARIABLE_VALUEB'_variables/9/.ATTRIBUTES/VARIABLE_VALUEB(_variables/10/.ATTRIBUTES/VARIABLE_VALUEB(_variables/11/.ATTRIBUTES/VARIABLE_VALUEB(_variables/12/.ATTRIBUTES/VARIABLE_VALUEB(_variables/13/.ATTRIBUTES/VARIABLE_VALUEB(_variables/14/.ATTRIBUTES/VARIABLE_VALUEB(_variables/15/.ATTRIBUTES/VARIABLE_VALUEB(_variables/16/.ATTRIBUTES/VARIABLE_VALUEB(_variables/17/.ATTRIBUTES/VARIABLE_VALUEB(_variables/18/.ATTRIBUTES/VARIABLE_VALUEB(_variables/19/.ATTRIBUTES/VARIABLE_VALUEB(_variables/20/.ATTRIBUTES/VARIABLE_VALUEB(_variables/21/.ATTRIBUTES/VARIABLE_VALUEB(_variables/22/.ATTRIBUTES/VARIABLE_VALUEB(_variables/23/.ATTRIBUTES/VARIABLE_VALUEB(_variables/24/.ATTRIBUTES/VARIABLE_VALUEB(_variables/25/.ATTRIBUTES/VARIABLE_VALUEB(_variables/26/.ATTRIBUTES/VARIABLE_VALUEB(_variables/27/.ATTRIBUTES/VARIABLE_VALUEB(_variables/28/.ATTRIBUTES/VARIABLE_VALUEB(_variables/29/.ATTRIBUTES/VARIABLE_VALUEB(_variables/30/.ATTRIBUTES/VARIABLE_VALUEB(_variables/31/.ATTRIBUTES/VARIABLE_VALUEB(_variables/32/.ATTRIBUTES/VARIABLE_VALUEB(_variables/33/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names╠
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesО
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:06savev2_control_network_normalize_b_read_readvariableop6savev2_control_network_normalize_w_read_readvariableop<savev2_control_network_layer_norm_offset_read_readvariableop;savev2_control_network_layer_norm_scale_read_readvariableopAsavev2_layernormandresidualmlp_mlp_linear_0_b_read_readvariableopAsavev2_layernormandresidualmlp_mlp_linear_0_w_read_readvariableopCsavev2_layernormandresidualmlp_mlp_linear_0_b_1_read_readvariableopCsavev2_layernormandresidualmlp_mlp_linear_0_w_1_read_readvariableopAsavev2_layernormandresidualmlp_mlp_linear_1_b_read_readvariableopAsavev2_layernormandresidualmlp_mlp_linear_1_w_read_readvariableop]savev2_layernormandresidualmlp_residuallayernormwrapper_layer_norm_offset_read_readvariableop\savev2_layernormandresidualmlp_residuallayernormwrapper_layer_norm_scale_read_readvariableopCsavev2_layernormandresidualmlp_mlp_linear_0_b_2_read_readvariableopCsavev2_layernormandresidualmlp_mlp_linear_0_w_2_read_readvariableopCsavev2_layernormandresidualmlp_mlp_linear_1_b_1_read_readvariableopCsavev2_layernormandresidualmlp_mlp_linear_1_w_1_read_readvariableop_savev2_layernormandresidualmlp_residuallayernormwrapper_layer_norm_offset_1_read_readvariableop^savev2_layernormandresidualmlp_residuallayernormwrapper_layer_norm_scale_1_read_readvariableopCsavev2_layernormandresidualmlp_mlp_linear_0_b_3_read_readvariableopCsavev2_layernormandresidualmlp_mlp_linear_0_w_3_read_readvariableopCsavev2_layernormandresidualmlp_mlp_linear_1_b_2_read_readvariableopCsavev2_layernormandresidualmlp_mlp_linear_1_w_2_read_readvariableop_savev2_layernormandresidualmlp_residuallayernormwrapper_layer_norm_offset_2_read_readvariableop^savev2_layernormandresidualmlp_residuallayernormwrapper_layer_norm_scale_2_read_readvariableopCsavev2_layernormandresidualmlp_mlp_linear_0_b_4_read_readvariableopCsavev2_layernormandresidualmlp_mlp_linear_0_w_4_read_readvariableopCsavev2_layernormandresidualmlp_mlp_linear_1_b_3_read_readvariableopCsavev2_layernormandresidualmlp_mlp_linear_1_w_3_read_readvariableop_savev2_layernormandresidualmlp_residuallayernormwrapper_layer_norm_offset_3_read_readvariableop^savev2_layernormandresidualmlp_residuallayernormwrapper_layer_norm_scale_3_read_readvariableop>savev2_multivariatenormaldiaghead_linear_b_read_readvariableop>savev2_multivariatenormaldiaghead_linear_w_read_readvariableop@savev2_multivariatenormaldiaghead_linear_b_1_read_readvariableop@savev2_multivariatenormaldiaghead_linear_w_1_read_readvariableop"/device:CPU:0*
_output_shapes
 *0
dtypes&
$2"2
SaveV2Г
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shardм
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1в
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_namesО
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slices╧
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1у
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesм
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

IdentityБ

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*╛
_input_shapesм
й: :м:	м:м:м:А:
мА:А:
АА:А:
АА:А:А:А:
АА:А:
АА:А:А:А:
АА:А:
АА:А:А:А:
АА:А:
АА:А:А::	А::	А: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:!

_output_shapes	
:м:%!

_output_shapes
:	м:!

_output_shapes	
:м:!

_output_shapes	
:м:!

_output_shapes	
:А:&"
 
_output_shapes
:
мА:!

_output_shapes	
:А:&"
 
_output_shapes
:
АА:!	

_output_shapes	
:А:&
"
 
_output_shapes
:
АА:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:&"
 
_output_shapes
:
АА:!

_output_shapes	
:А:&"
 
_output_shapes
:
АА:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:&"
 
_output_shapes
:
АА:!

_output_shapes	
:А:&"
 
_output_shapes
:
АА:!

_output_shapes	
:А:!

_output_shapes	
:А:!

_output_shapes	
:А:&"
 
_output_shapes
:
АА:!

_output_shapes	
:А:&"
 
_output_shapes
:
АА:!

_output_shapes	
:А:!

_output_shapes	
:А: 

_output_shapes
::% !

_output_shapes
:	А: !

_output_shapes
::%"!

_output_shapes
:	А:#

_output_shapes
: "аJ
saver_filename:0StatefulPartitionedCall:0StatefulPartitionedCall_18"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp:╜"


_variables
_trainable_variables

signatures
&__call__
'_module
(initial_state"
acme_snapshot
з
0
1
2
3
4
	5

6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
 28
!29
"30
#31
$32
%33"
trackable_tuple_wrapper
з
0
1
2
3
4
	5

6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
 28
!29
"30
#31
$32
%33"
trackable_tuple_wrapper
"
signature_map
.:,м (2control_network/normalize/b
2:0	м (2control_network/normalize/w
4:2м (2!control_network/layer_norm/offset
3:1м (2 control_network/layer_norm/scale
9:7А (2&LayerNormAndResidualMLP/mlp/linear_0/b
>:<
мА (2&LayerNormAndResidualMLP/mlp/linear_0/w
9:7А (2&LayerNormAndResidualMLP/mlp/linear_0/b
>:<
АА (2&LayerNormAndResidualMLP/mlp/linear_0/w
9:7А (2&LayerNormAndResidualMLP/mlp/linear_1/b
>:<
АА (2&LayerNormAndResidualMLP/mlp/linear_1/w
U:SА (2BLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/offset
T:RА (2ALayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/scale
9:7А (2&LayerNormAndResidualMLP/mlp/linear_0/b
>:<
АА (2&LayerNormAndResidualMLP/mlp/linear_0/w
9:7А (2&LayerNormAndResidualMLP/mlp/linear_1/b
>:<
АА (2&LayerNormAndResidualMLP/mlp/linear_1/w
U:SА (2BLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/offset
T:RА (2ALayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/scale
9:7А (2&LayerNormAndResidualMLP/mlp/linear_0/b
>:<
АА (2&LayerNormAndResidualMLP/mlp/linear_0/w
9:7А (2&LayerNormAndResidualMLP/mlp/linear_1/b
>:<
АА (2&LayerNormAndResidualMLP/mlp/linear_1/w
U:SА (2BLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/offset
T:RА (2ALayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/scale
9:7А (2&LayerNormAndResidualMLP/mlp/linear_0/b
>:<
АА (2&LayerNormAndResidualMLP/mlp/linear_0/w
9:7А (2&LayerNormAndResidualMLP/mlp/linear_1/b
>:<
АА (2&LayerNormAndResidualMLP/mlp/linear_1/w
U:SА (2BLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/offset
T:RА (2ALayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/scale
5:3 (2#MultivariateNormalDiagHead/linear/b
::8	А (2#MultivariateNormalDiagHead/linear/w
5:3 (2#MultivariateNormalDiagHead/linear/b
::8	А (2#MultivariateNormalDiagHead/linear/w
─2┴
__inference___call___82212в
Щ▓Х
FullArgSpec
argsЪ
jself
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
└2╜
 __inference_wrapped_module_82517Ш
С▓Н
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╔2╞
__inference_initial_state_82520в
Щ▓Х
FullArgSpec
argsЪ
jself
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 ё
__inference___call___82212╥"	
! #"%$ в√
євя
тк▐
=
joint_angles-К*
args_0/joint_angles         
1
target'К$
args_0/target         
3
upright(К%
args_0/upright         
5
velocity)К&
args_0/velocity         
в
в 
к "*в'
К
0         
в
в M
__inference_initial_state_82520*в
в
К
args_0 
к "в
в ў
 __inference_wrapped_module_82517╥"	
! #"%$ в√
євя
тк▐
=
joint_angles-К*
args_0/joint_angles         
1
target'К$
args_0/target         
3
upright(К%
args_0/upright         
5
velocity)К&
args_0/velocity         
в
в 
к "*в'
К
0         
в
в 