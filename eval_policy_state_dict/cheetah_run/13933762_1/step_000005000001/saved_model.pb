ů
żŁ
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
ž
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
 "serve*2.2.02unknown8ĂŇ

control_network/normalize/bVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ź*,
shared_namecontrol_network/normalize/b

/control_network/normalize/b/Read/ReadVariableOpReadVariableOpcontrol_network/normalize/b*
_output_shapes	
:Ź*
dtype0

control_network/normalize/wVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ź*,
shared_namecontrol_network/normalize/w

/control_network/normalize/w/Read/ReadVariableOpReadVariableOpcontrol_network/normalize/w*
_output_shapes
:	Ź*
dtype0

!control_network/layer_norm/offsetVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ź*2
shared_name#!control_network/layer_norm/offset

5control_network/layer_norm/offset/Read/ReadVariableOpReadVariableOp!control_network/layer_norm/offset*
_output_shapes	
:Ź*
dtype0

 control_network/layer_norm/scaleVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ź*1
shared_name" control_network/layer_norm/scale

4control_network/layer_norm/scale/Read/ReadVariableOpReadVariableOp control_network/layer_norm/scale*
_output_shapes	
:Ź*
dtype0
Ľ
&LayerNormAndResidualMLP/mlp/linear_0/bVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&LayerNormAndResidualMLP/mlp/linear_0/b

:LayerNormAndResidualMLP/mlp/linear_0/b/Read/ReadVariableOpReadVariableOp&LayerNormAndResidualMLP/mlp/linear_0/b*
_output_shapes	
:*
dtype0
Ş
&LayerNormAndResidualMLP/mlp/linear_0/wVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Ź*7
shared_name(&LayerNormAndResidualMLP/mlp/linear_0/w
Ł
:LayerNormAndResidualMLP/mlp/linear_0/w/Read/ReadVariableOpReadVariableOp&LayerNormAndResidualMLP/mlp/linear_0/w* 
_output_shapes
:
Ź*
dtype0
Š
(LayerNormAndResidualMLP/mlp/linear_0/b_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(LayerNormAndResidualMLP/mlp/linear_0/b_1
˘
<LayerNormAndResidualMLP/mlp/linear_0/b_1/Read/ReadVariableOpReadVariableOp(LayerNormAndResidualMLP/mlp/linear_0/b_1*
_output_shapes	
:*
dtype0
Ž
(LayerNormAndResidualMLP/mlp/linear_0/w_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:
*9
shared_name*(LayerNormAndResidualMLP/mlp/linear_0/w_1
§
<LayerNormAndResidualMLP/mlp/linear_0/w_1/Read/ReadVariableOpReadVariableOp(LayerNormAndResidualMLP/mlp/linear_0/w_1* 
_output_shapes
:
*
dtype0
Ľ
&LayerNormAndResidualMLP/mlp/linear_1/bVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&LayerNormAndResidualMLP/mlp/linear_1/b

:LayerNormAndResidualMLP/mlp/linear_1/b/Read/ReadVariableOpReadVariableOp&LayerNormAndResidualMLP/mlp/linear_1/b*
_output_shapes	
:*
dtype0
Ş
&LayerNormAndResidualMLP/mlp/linear_1/wVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*7
shared_name(&LayerNormAndResidualMLP/mlp/linear_1/w
Ł
:LayerNormAndResidualMLP/mlp/linear_1/w/Read/ReadVariableOpReadVariableOp&LayerNormAndResidualMLP/mlp/linear_1/w* 
_output_shapes
:
*
dtype0
Ý
BLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/offsetVarHandleOp*
_output_shapes
: *
dtype0*
shape:*S
shared_nameDBLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/offset
Ö
VLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/offset/Read/ReadVariableOpReadVariableOpBLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/offset*
_output_shapes	
:*
dtype0
Ű
ALayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/scaleVarHandleOp*
_output_shapes
: *
dtype0*
shape:*R
shared_nameCALayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/scale
Ô
ULayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/scale/Read/ReadVariableOpReadVariableOpALayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/scale*
_output_shapes	
:*
dtype0
Š
(LayerNormAndResidualMLP/mlp/linear_0/b_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(LayerNormAndResidualMLP/mlp/linear_0/b_2
˘
<LayerNormAndResidualMLP/mlp/linear_0/b_2/Read/ReadVariableOpReadVariableOp(LayerNormAndResidualMLP/mlp/linear_0/b_2*
_output_shapes	
:*
dtype0
Ž
(LayerNormAndResidualMLP/mlp/linear_0/w_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:
*9
shared_name*(LayerNormAndResidualMLP/mlp/linear_0/w_2
§
<LayerNormAndResidualMLP/mlp/linear_0/w_2/Read/ReadVariableOpReadVariableOp(LayerNormAndResidualMLP/mlp/linear_0/w_2* 
_output_shapes
:
*
dtype0
Š
(LayerNormAndResidualMLP/mlp/linear_1/b_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(LayerNormAndResidualMLP/mlp/linear_1/b_1
˘
<LayerNormAndResidualMLP/mlp/linear_1/b_1/Read/ReadVariableOpReadVariableOp(LayerNormAndResidualMLP/mlp/linear_1/b_1*
_output_shapes	
:*
dtype0
Ž
(LayerNormAndResidualMLP/mlp/linear_1/w_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:
*9
shared_name*(LayerNormAndResidualMLP/mlp/linear_1/w_1
§
<LayerNormAndResidualMLP/mlp/linear_1/w_1/Read/ReadVariableOpReadVariableOp(LayerNormAndResidualMLP/mlp/linear_1/w_1* 
_output_shapes
:
*
dtype0
á
DLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/offset_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*U
shared_nameFDLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/offset_1
Ú
XLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/offset_1/Read/ReadVariableOpReadVariableOpDLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/offset_1*
_output_shapes	
:*
dtype0
ß
CLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/scale_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*T
shared_nameECLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/scale_1
Ř
WLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/scale_1/Read/ReadVariableOpReadVariableOpCLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/scale_1*
_output_shapes	
:*
dtype0
Š
(LayerNormAndResidualMLP/mlp/linear_0/b_3VarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(LayerNormAndResidualMLP/mlp/linear_0/b_3
˘
<LayerNormAndResidualMLP/mlp/linear_0/b_3/Read/ReadVariableOpReadVariableOp(LayerNormAndResidualMLP/mlp/linear_0/b_3*
_output_shapes	
:*
dtype0
Ž
(LayerNormAndResidualMLP/mlp/linear_0/w_3VarHandleOp*
_output_shapes
: *
dtype0*
shape:
*9
shared_name*(LayerNormAndResidualMLP/mlp/linear_0/w_3
§
<LayerNormAndResidualMLP/mlp/linear_0/w_3/Read/ReadVariableOpReadVariableOp(LayerNormAndResidualMLP/mlp/linear_0/w_3* 
_output_shapes
:
*
dtype0
Š
(LayerNormAndResidualMLP/mlp/linear_1/b_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(LayerNormAndResidualMLP/mlp/linear_1/b_2
˘
<LayerNormAndResidualMLP/mlp/linear_1/b_2/Read/ReadVariableOpReadVariableOp(LayerNormAndResidualMLP/mlp/linear_1/b_2*
_output_shapes	
:*
dtype0
Ž
(LayerNormAndResidualMLP/mlp/linear_1/w_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:
*9
shared_name*(LayerNormAndResidualMLP/mlp/linear_1/w_2
§
<LayerNormAndResidualMLP/mlp/linear_1/w_2/Read/ReadVariableOpReadVariableOp(LayerNormAndResidualMLP/mlp/linear_1/w_2* 
_output_shapes
:
*
dtype0
á
DLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/offset_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:*U
shared_nameFDLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/offset_2
Ú
XLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/offset_2/Read/ReadVariableOpReadVariableOpDLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/offset_2*
_output_shapes	
:*
dtype0
ß
CLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/scale_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:*T
shared_nameECLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/scale_2
Ř
WLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/scale_2/Read/ReadVariableOpReadVariableOpCLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/scale_2*
_output_shapes	
:*
dtype0
Š
(LayerNormAndResidualMLP/mlp/linear_0/b_4VarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(LayerNormAndResidualMLP/mlp/linear_0/b_4
˘
<LayerNormAndResidualMLP/mlp/linear_0/b_4/Read/ReadVariableOpReadVariableOp(LayerNormAndResidualMLP/mlp/linear_0/b_4*
_output_shapes	
:*
dtype0
Ž
(LayerNormAndResidualMLP/mlp/linear_0/w_4VarHandleOp*
_output_shapes
: *
dtype0*
shape:
*9
shared_name*(LayerNormAndResidualMLP/mlp/linear_0/w_4
§
<LayerNormAndResidualMLP/mlp/linear_0/w_4/Read/ReadVariableOpReadVariableOp(LayerNormAndResidualMLP/mlp/linear_0/w_4* 
_output_shapes
:
*
dtype0
Š
(LayerNormAndResidualMLP/mlp/linear_1/b_3VarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(LayerNormAndResidualMLP/mlp/linear_1/b_3
˘
<LayerNormAndResidualMLP/mlp/linear_1/b_3/Read/ReadVariableOpReadVariableOp(LayerNormAndResidualMLP/mlp/linear_1/b_3*
_output_shapes	
:*
dtype0
Ž
(LayerNormAndResidualMLP/mlp/linear_1/w_3VarHandleOp*
_output_shapes
: *
dtype0*
shape:
*9
shared_name*(LayerNormAndResidualMLP/mlp/linear_1/w_3
§
<LayerNormAndResidualMLP/mlp/linear_1/w_3/Read/ReadVariableOpReadVariableOp(LayerNormAndResidualMLP/mlp/linear_1/w_3* 
_output_shapes
:
*
dtype0
á
DLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/offset_3VarHandleOp*
_output_shapes
: *
dtype0*
shape:*U
shared_nameFDLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/offset_3
Ú
XLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/offset_3/Read/ReadVariableOpReadVariableOpDLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/offset_3*
_output_shapes	
:*
dtype0
ß
CLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/scale_3VarHandleOp*
_output_shapes
: *
dtype0*
shape:*T
shared_nameECLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/scale_3
Ř
WLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/scale_3/Read/ReadVariableOpReadVariableOpCLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/scale_3*
_output_shapes	
:*
dtype0

#MultivariateNormalDiagHead/linear/bVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#MultivariateNormalDiagHead/linear/b

7MultivariateNormalDiagHead/linear/b/Read/ReadVariableOpReadVariableOp#MultivariateNormalDiagHead/linear/b*
_output_shapes
:*
dtype0
Ł
#MultivariateNormalDiagHead/linear/wVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*4
shared_name%#MultivariateNormalDiagHead/linear/w

7MultivariateNormalDiagHead/linear/w/Read/ReadVariableOpReadVariableOp#MultivariateNormalDiagHead/linear/w*
_output_shapes
:	*
dtype0
˘
%MultivariateNormalDiagHead/linear/b_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%MultivariateNormalDiagHead/linear/b_1

9MultivariateNormalDiagHead/linear/b_1/Read/ReadVariableOpReadVariableOp%MultivariateNormalDiagHead/linear/b_1*
_output_shapes
:*
dtype0
§
%MultivariateNormalDiagHead/linear/w_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:	*6
shared_name'%MultivariateNormalDiagHead/linear/w_1
 
9MultivariateNormalDiagHead/linear/w_1/Read/ReadVariableOpReadVariableOp%MultivariateNormalDiagHead/linear/w_1*
_output_shapes
:	*
dtype0

NoOpNoOp
"
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ë!
valueÁ!Bž! Bˇ!
:

_variables
_trainable_variables

signatures

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

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
~
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

VARIABLE_VALUEDLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/offset_1(_variables/16/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUECLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/scale_1(_variables/17/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUE(LayerNormAndResidualMLP/mlp/linear_0/b_3(_variables/18/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUE(LayerNormAndResidualMLP/mlp/linear_0/w_3(_variables/19/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUE(LayerNormAndResidualMLP/mlp/linear_1/b_2(_variables/20/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUE(LayerNormAndResidualMLP/mlp/linear_1/w_2(_variables/21/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEDLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/offset_2(_variables/22/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUECLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/scale_2(_variables/23/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUE(LayerNormAndResidualMLP/mlp/linear_0/b_4(_variables/24/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUE(LayerNormAndResidualMLP/mlp/linear_0/w_4(_variables/25/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUE(LayerNormAndResidualMLP/mlp/linear_1/b_3(_variables/26/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUE(LayerNormAndResidualMLP/mlp/linear_1/w_3(_variables/27/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEDLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/offset_3(_variables/28/.ATTRIBUTES/VARIABLE_VALUE

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
Ś
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

TPU_SYSTEM2J 8**
f%R#
!__inference__traced_save_22309538
ű
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

TPU_SYSTEM2J 8*-
f(R&
$__inference__traced_restore_22309652­

.
 __inference_initial_state_248504

args_0*
_input_shapes
: :> :

_output_shapes
: 
 
_user_specified_nameargs_0
Z
Í
!__inference__traced_save_22309538
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

identity_1˘MergeV2Checkpoints˘SaveV2˘SaveV2_1
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
value3B1 B+_temp_cf2a928fd9924ac0be788690733b7248/part2	
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
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardŚ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*
valueB"B'_variables/0/.ATTRIBUTES/VARIABLE_VALUEB'_variables/1/.ATTRIBUTES/VARIABLE_VALUEB'_variables/2/.ATTRIBUTES/VARIABLE_VALUEB'_variables/3/.ATTRIBUTES/VARIABLE_VALUEB'_variables/4/.ATTRIBUTES/VARIABLE_VALUEB'_variables/5/.ATTRIBUTES/VARIABLE_VALUEB'_variables/6/.ATTRIBUTES/VARIABLE_VALUEB'_variables/7/.ATTRIBUTES/VARIABLE_VALUEB'_variables/8/.ATTRIBUTES/VARIABLE_VALUEB'_variables/9/.ATTRIBUTES/VARIABLE_VALUEB(_variables/10/.ATTRIBUTES/VARIABLE_VALUEB(_variables/11/.ATTRIBUTES/VARIABLE_VALUEB(_variables/12/.ATTRIBUTES/VARIABLE_VALUEB(_variables/13/.ATTRIBUTES/VARIABLE_VALUEB(_variables/14/.ATTRIBUTES/VARIABLE_VALUEB(_variables/15/.ATTRIBUTES/VARIABLE_VALUEB(_variables/16/.ATTRIBUTES/VARIABLE_VALUEB(_variables/17/.ATTRIBUTES/VARIABLE_VALUEB(_variables/18/.ATTRIBUTES/VARIABLE_VALUEB(_variables/19/.ATTRIBUTES/VARIABLE_VALUEB(_variables/20/.ATTRIBUTES/VARIABLE_VALUEB(_variables/21/.ATTRIBUTES/VARIABLE_VALUEB(_variables/22/.ATTRIBUTES/VARIABLE_VALUEB(_variables/23/.ATTRIBUTES/VARIABLE_VALUEB(_variables/24/.ATTRIBUTES/VARIABLE_VALUEB(_variables/25/.ATTRIBUTES/VARIABLE_VALUEB(_variables/26/.ATTRIBUTES/VARIABLE_VALUEB(_variables/27/.ATTRIBUTES/VARIABLE_VALUEB(_variables/28/.ATTRIBUTES/VARIABLE_VALUEB(_variables/29/.ATTRIBUTES/VARIABLE_VALUEB(_variables/30/.ATTRIBUTES/VARIABLE_VALUEB(_variables/31/.ATTRIBUTES/VARIABLE_VALUEB(_variables/32/.ATTRIBUTES/VARIABLE_VALUEB(_variables/33/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_namesĚ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:06savev2_control_network_normalize_b_read_readvariableop6savev2_control_network_normalize_w_read_readvariableop<savev2_control_network_layer_norm_offset_read_readvariableop;savev2_control_network_layer_norm_scale_read_readvariableopAsavev2_layernormandresidualmlp_mlp_linear_0_b_read_readvariableopAsavev2_layernormandresidualmlp_mlp_linear_0_w_read_readvariableopCsavev2_layernormandresidualmlp_mlp_linear_0_b_1_read_readvariableopCsavev2_layernormandresidualmlp_mlp_linear_0_w_1_read_readvariableopAsavev2_layernormandresidualmlp_mlp_linear_1_b_read_readvariableopAsavev2_layernormandresidualmlp_mlp_linear_1_w_read_readvariableop]savev2_layernormandresidualmlp_residuallayernormwrapper_layer_norm_offset_read_readvariableop\savev2_layernormandresidualmlp_residuallayernormwrapper_layer_norm_scale_read_readvariableopCsavev2_layernormandresidualmlp_mlp_linear_0_b_2_read_readvariableopCsavev2_layernormandresidualmlp_mlp_linear_0_w_2_read_readvariableopCsavev2_layernormandresidualmlp_mlp_linear_1_b_1_read_readvariableopCsavev2_layernormandresidualmlp_mlp_linear_1_w_1_read_readvariableop_savev2_layernormandresidualmlp_residuallayernormwrapper_layer_norm_offset_1_read_readvariableop^savev2_layernormandresidualmlp_residuallayernormwrapper_layer_norm_scale_1_read_readvariableopCsavev2_layernormandresidualmlp_mlp_linear_0_b_3_read_readvariableopCsavev2_layernormandresidualmlp_mlp_linear_0_w_3_read_readvariableopCsavev2_layernormandresidualmlp_mlp_linear_1_b_2_read_readvariableopCsavev2_layernormandresidualmlp_mlp_linear_1_w_2_read_readvariableop_savev2_layernormandresidualmlp_residuallayernormwrapper_layer_norm_offset_2_read_readvariableop^savev2_layernormandresidualmlp_residuallayernormwrapper_layer_norm_scale_2_read_readvariableopCsavev2_layernormandresidualmlp_mlp_linear_0_b_4_read_readvariableopCsavev2_layernormandresidualmlp_mlp_linear_0_w_4_read_readvariableopCsavev2_layernormandresidualmlp_mlp_linear_1_b_3_read_readvariableopCsavev2_layernormandresidualmlp_mlp_linear_1_w_3_read_readvariableop_savev2_layernormandresidualmlp_residuallayernormwrapper_layer_norm_offset_3_read_readvariableop^savev2_layernormandresidualmlp_residuallayernormwrapper_layer_norm_scale_3_read_readvariableop>savev2_multivariatenormaldiaghead_linear_b_read_readvariableop>savev2_multivariatenormaldiaghead_linear_w_read_readvariableop@savev2_multivariatenormaldiaghead_linear_b_1_read_readvariableop@savev2_multivariatenormaldiaghead_linear_w_1_read_readvariableop"/device:CPU:0*
_output_shapes
 *0
dtypes&
$2"2
SaveV2
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shardŹ
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1˘
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slicesĎ
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1ă
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesŹ
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*ž
_input_shapesŹ
Š: :Ź:	Ź:Ź:Ź::
Ź::
::
::::
::
::::
::
::::
::
::::	::	: 2(
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
:Ź:%!

_output_shapes
:	Ź:!

_output_shapes	
:Ź:!

_output_shapes	
:Ź:!

_output_shapes	
::&"
 
_output_shapes
:
Ź:!

_output_shapes	
::&"
 
_output_shapes
:
:!	

_output_shapes	
::&
"
 
_output_shapes
:
:!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::!

_output_shapes	
:: 

_output_shapes
::% !

_output_shapes
:	: !

_output_shapes
::%"!

_output_shapes
:	:#

_output_shapes
: 


__inference___call___248216
args_0_position
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
identity˘StatefulPartitionedCallˇ
StatefulPartitionedCallStatefulPartitionedCallargs_0_positionargs_0_velocityunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_32*/
Tin(
&2$*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*D
_read_only_resource_inputs&
$"	
 !"#*C
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
__inference_wrapped_module_88322
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*Ă
_input_shapesą
Ž:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙	::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_nameargs_0/position:XT
'
_output_shapes
:˙˙˙˙˙˙˙˙˙	
)
_user_specified_nameargs_0/velocity:
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
: :#

_output_shapes
: 
Ű

$__inference__traced_restore_22309652
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
identity_35˘AssignVariableOp˘AssignVariableOp_1˘AssignVariableOp_10˘AssignVariableOp_11˘AssignVariableOp_12˘AssignVariableOp_13˘AssignVariableOp_14˘AssignVariableOp_15˘AssignVariableOp_16˘AssignVariableOp_17˘AssignVariableOp_18˘AssignVariableOp_19˘AssignVariableOp_2˘AssignVariableOp_20˘AssignVariableOp_21˘AssignVariableOp_22˘AssignVariableOp_23˘AssignVariableOp_24˘AssignVariableOp_25˘AssignVariableOp_26˘AssignVariableOp_27˘AssignVariableOp_28˘AssignVariableOp_29˘AssignVariableOp_3˘AssignVariableOp_30˘AssignVariableOp_31˘AssignVariableOp_32˘AssignVariableOp_33˘AssignVariableOp_4˘AssignVariableOp_5˘AssignVariableOp_6˘AssignVariableOp_7˘AssignVariableOp_8˘AssignVariableOp_9˘	RestoreV2˘RestoreV2_1
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*
valueB"B'_variables/0/.ATTRIBUTES/VARIABLE_VALUEB'_variables/1/.ATTRIBUTES/VARIABLE_VALUEB'_variables/2/.ATTRIBUTES/VARIABLE_VALUEB'_variables/3/.ATTRIBUTES/VARIABLE_VALUEB'_variables/4/.ATTRIBUTES/VARIABLE_VALUEB'_variables/5/.ATTRIBUTES/VARIABLE_VALUEB'_variables/6/.ATTRIBUTES/VARIABLE_VALUEB'_variables/7/.ATTRIBUTES/VARIABLE_VALUEB'_variables/8/.ATTRIBUTES/VARIABLE_VALUEB'_variables/9/.ATTRIBUTES/VARIABLE_VALUEB(_variables/10/.ATTRIBUTES/VARIABLE_VALUEB(_variables/11/.ATTRIBUTES/VARIABLE_VALUEB(_variables/12/.ATTRIBUTES/VARIABLE_VALUEB(_variables/13/.ATTRIBUTES/VARIABLE_VALUEB(_variables/14/.ATTRIBUTES/VARIABLE_VALUEB(_variables/15/.ATTRIBUTES/VARIABLE_VALUEB(_variables/16/.ATTRIBUTES/VARIABLE_VALUEB(_variables/17/.ATTRIBUTES/VARIABLE_VALUEB(_variables/18/.ATTRIBUTES/VARIABLE_VALUEB(_variables/19/.ATTRIBUTES/VARIABLE_VALUEB(_variables/20/.ATTRIBUTES/VARIABLE_VALUEB(_variables/21/.ATTRIBUTES/VARIABLE_VALUEB(_variables/22/.ATTRIBUTES/VARIABLE_VALUEB(_variables/23/.ATTRIBUTES/VARIABLE_VALUEB(_variables/24/.ATTRIBUTES/VARIABLE_VALUEB(_variables/25/.ATTRIBUTES/VARIABLE_VALUEB(_variables/26/.ATTRIBUTES/VARIABLE_VALUEB(_variables/27/.ATTRIBUTES/VARIABLE_VALUEB(_variables/28/.ATTRIBUTES/VARIABLE_VALUEB(_variables/29/.ATTRIBUTES/VARIABLE_VALUEB(_variables/30/.ATTRIBUTES/VARIABLE_VALUEB(_variables/31/.ATTRIBUTES/VARIABLE_VALUEB(_variables/32/.ATTRIBUTES/VARIABLE_VALUEB(_variables/33/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_namesŇ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesŘ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapes
::::::::::::::::::::::::::::::::::*0
dtypes&
$2"2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOp,assignvariableop_control_network_normalize_bIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1¤
AssignVariableOp_1AssignVariableOp.assignvariableop_1_control_network_normalize_wIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2Ş
AssignVariableOp_2AssignVariableOp4assignvariableop_2_control_network_layer_norm_offsetIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3Š
AssignVariableOp_3AssignVariableOp3assignvariableop_3_control_network_layer_norm_scaleIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4Ż
AssignVariableOp_4AssignVariableOp9assignvariableop_4_layernormandresidualmlp_mlp_linear_0_bIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5Ż
AssignVariableOp_5AssignVariableOp9assignvariableop_5_layernormandresidualmlp_mlp_linear_0_wIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6ą
AssignVariableOp_6AssignVariableOp;assignvariableop_6_layernormandresidualmlp_mlp_linear_0_b_1Identity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7ą
AssignVariableOp_7AssignVariableOp;assignvariableop_7_layernormandresidualmlp_mlp_linear_0_w_1Identity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8Ż
AssignVariableOp_8AssignVariableOp9assignvariableop_8_layernormandresidualmlp_mlp_linear_1_bIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9Ż
AssignVariableOp_9AssignVariableOp9assignvariableop_9_layernormandresidualmlp_mlp_linear_1_wIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10Ď
AssignVariableOp_10AssignVariableOpVassignvariableop_10_layernormandresidualmlp_residuallayernormwrapper_layer_norm_offsetIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11Î
AssignVariableOp_11AssignVariableOpUassignvariableop_11_layernormandresidualmlp_residuallayernormwrapper_layer_norm_scaleIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12ľ
AssignVariableOp_12AssignVariableOp<assignvariableop_12_layernormandresidualmlp_mlp_linear_0_b_2Identity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13ľ
AssignVariableOp_13AssignVariableOp<assignvariableop_13_layernormandresidualmlp_mlp_linear_0_w_2Identity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14ľ
AssignVariableOp_14AssignVariableOp<assignvariableop_14_layernormandresidualmlp_mlp_linear_1_b_1Identity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15ľ
AssignVariableOp_15AssignVariableOp<assignvariableop_15_layernormandresidualmlp_mlp_linear_1_w_1Identity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16Ń
AssignVariableOp_16AssignVariableOpXassignvariableop_16_layernormandresidualmlp_residuallayernormwrapper_layer_norm_offset_1Identity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17Đ
AssignVariableOp_17AssignVariableOpWassignvariableop_17_layernormandresidualmlp_residuallayernormwrapper_layer_norm_scale_1Identity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18ľ
AssignVariableOp_18AssignVariableOp<assignvariableop_18_layernormandresidualmlp_mlp_linear_0_b_3Identity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19ľ
AssignVariableOp_19AssignVariableOp<assignvariableop_19_layernormandresidualmlp_mlp_linear_0_w_3Identity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:2
Identity_20ľ
AssignVariableOp_20AssignVariableOp<assignvariableop_20_layernormandresidualmlp_mlp_linear_1_b_2Identity_20:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21ľ
AssignVariableOp_21AssignVariableOp<assignvariableop_21_layernormandresidualmlp_mlp_linear_1_w_2Identity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21_
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:2
Identity_22Ń
AssignVariableOp_22AssignVariableOpXassignvariableop_22_layernormandresidualmlp_residuallayernormwrapper_layer_norm_offset_2Identity_22:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_22_
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:2
Identity_23Đ
AssignVariableOp_23AssignVariableOpWassignvariableop_23_layernormandresidualmlp_residuallayernormwrapper_layer_norm_scale_2Identity_23:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_23_
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:2
Identity_24ľ
AssignVariableOp_24AssignVariableOp<assignvariableop_24_layernormandresidualmlp_mlp_linear_0_b_4Identity_24:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_24_
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:2
Identity_25ľ
AssignVariableOp_25AssignVariableOp<assignvariableop_25_layernormandresidualmlp_mlp_linear_0_w_4Identity_25:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_25_
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:2
Identity_26ľ
AssignVariableOp_26AssignVariableOp<assignvariableop_26_layernormandresidualmlp_mlp_linear_1_b_3Identity_26:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_26_
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:2
Identity_27ľ
AssignVariableOp_27AssignVariableOp<assignvariableop_27_layernormandresidualmlp_mlp_linear_1_w_3Identity_27:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_27_
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:2
Identity_28Ń
AssignVariableOp_28AssignVariableOpXassignvariableop_28_layernormandresidualmlp_residuallayernormwrapper_layer_norm_offset_3Identity_28:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_28_
Identity_29IdentityRestoreV2:tensors:29*
T0*
_output_shapes
:2
Identity_29Đ
AssignVariableOp_29AssignVariableOpWassignvariableop_29_layernormandresidualmlp_residuallayernormwrapper_layer_norm_scale_3Identity_29:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_29_
Identity_30IdentityRestoreV2:tensors:30*
T0*
_output_shapes
:2
Identity_30°
AssignVariableOp_30AssignVariableOp7assignvariableop_30_multivariatenormaldiaghead_linear_bIdentity_30:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_30_
Identity_31IdentityRestoreV2:tensors:31*
T0*
_output_shapes
:2
Identity_31°
AssignVariableOp_31AssignVariableOp7assignvariableop_31_multivariatenormaldiaghead_linear_wIdentity_31:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_31_
Identity_32IdentityRestoreV2:tensors:32*
T0*
_output_shapes
:2
Identity_32˛
AssignVariableOp_32AssignVariableOp9assignvariableop_32_multivariatenormaldiaghead_linear_b_1Identity_32:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_32_
Identity_33IdentityRestoreV2:tensors:33*
T0*
_output_shapes
:2
Identity_33˛
AssignVariableOp_33AssignVariableOp9assignvariableop_33_multivariatenormaldiaghead_linear_w_1Identity_33:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_33¨
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slicesÄ
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
NoOpĘ
Identity_34Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_34×
Identity_35IdentityIdentity_34:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_35"#
identity_35Identity_35:output:0*
_input_shapes
: ::::::::::::::::::::::::::::::::::2$
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

ç
__inference_wrapped_module_8832

args_0
args_0_1<
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
identityt
control_network/flatten/ShapeShapeargs_0*
T0*
_output_shapes
:2
control_network/flatten/Shape¤
+control_network/flatten/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+control_network/flatten/strided_slice/stack¨
-control_network/flatten/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-control_network/flatten/strided_slice/stack_1¨
-control_network/flatten/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-control_network/flatten/strided_slice/stack_2đ
%control_network/flatten/strided_sliceStridedSlice&control_network/flatten/Shape:output:04control_network/flatten/strided_slice/stack:output:06control_network/flatten/strided_slice/stack_1:output:06control_network/flatten/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2'
%control_network/flatten/strided_slice
'control_network/flatten/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'control_network/flatten/concat/values_1
#control_network/flatten/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#control_network/flatten/concat/axis
control_network/flatten/concatConcatV2.control_network/flatten/strided_slice:output:00control_network/flatten/concat/values_1:output:0,control_network/flatten/concat/axis:output:0*
N*
T0*
_output_shapes
:2 
control_network/flatten/concat°
control_network/flatten/ReshapeReshapeargs_0'control_network/flatten/concat:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2!
control_network/flatten/Reshapez
control_network/flatten_1/ShapeShapeargs_0_1*
T0*
_output_shapes
:2!
control_network/flatten_1/Shape¨
-control_network/flatten_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-control_network/flatten_1/strided_slice/stackŹ
/control_network/flatten_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/control_network/flatten_1/strided_slice/stack_1Ź
/control_network/flatten_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/control_network/flatten_1/strided_slice/stack_2ü
'control_network/flatten_1/strided_sliceStridedSlice(control_network/flatten_1/Shape:output:06control_network/flatten_1/strided_slice/stack:output:08control_network/flatten_1/strided_slice/stack_1:output:08control_network/flatten_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2)
'control_network/flatten_1/strided_slice 
)control_network/flatten_1/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:	2+
)control_network/flatten_1/concat/values_1
%control_network/flatten_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2'
%control_network/flatten_1/concat/axis
 control_network/flatten_1/concatConcatV20control_network/flatten_1/strided_slice:output:02control_network/flatten_1/concat/values_1:output:0.control_network/flatten_1/concat/axis:output:0*
N*
T0*
_output_shapes
:2"
 control_network/flatten_1/concat¸
!control_network/flatten_1/ReshapeReshapeargs_0_1)control_network/flatten_1/concat:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙	2#
!control_network/flatten_1/Reshape|
control_network/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
control_network/concat/axisó
control_network/concatConcatV2(control_network/flatten/Reshape:output:0*control_network/flatten_1/Reshape:output:0$control_network/concat/axis:output:0*
N*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
control_network/concatÜ
/control_network/normalize/MatMul/ReadVariableOpReadVariableOp8control_network_normalize_matmul_readvariableop_resource*
_output_shapes
:	Ź*
dtype021
/control_network/normalize/MatMul/ReadVariableOpŰ
 control_network/normalize/MatMulMatMulcontrol_network/concat:output:07control_network/normalize/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź2"
 control_network/normalize/MatMulĎ
,control_network/normalize/Add/ReadVariableOpReadVariableOp5control_network_normalize_add_readvariableop_resource*
_output_shapes	
:Ź*
dtype02.
,control_network/normalize/Add/ReadVariableOpÚ
control_network/normalize/AddAdd*control_network/normalize/MatMul:product:04control_network/normalize/Add/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź2
control_network/normalize/AddŔ
9control_network/layer_norm/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9control_network/layer_norm/moments/mean/reduction_indices
'control_network/layer_norm/moments/meanMean!control_network/normalize/Add:z:0Bcontrol_network/layer_norm/moments/mean/reduction_indices:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	keep_dims(2)
'control_network/layer_norm/moments/meanÖ
/control_network/layer_norm/moments/StopGradientStopGradient0control_network/layer_norm/moments/mean:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙21
/control_network/layer_norm/moments/StopGradient
4control_network/layer_norm/moments/SquaredDifferenceSquaredDifference!control_network/normalize/Add:z:08control_network/layer_norm/moments/StopGradient:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź26
4control_network/layer_norm/moments/SquaredDifferenceČ
=control_network/layer_norm/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2?
=control_network/layer_norm/moments/variance/reduction_indices§
+control_network/layer_norm/moments/varianceMean8control_network/layer_norm/moments/SquaredDifference:z:0Fcontrol_network/layer_norm/moments/variance/reduction_indices:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	keep_dims(2-
+control_network/layer_norm/moments/variance
*control_network/layer_norm/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŹĹ'72,
*control_network/layer_norm/batchnorm/add/yú
(control_network/layer_norm/batchnorm/addAddV24control_network/layer_norm/moments/variance:output:03control_network/layer_norm/batchnorm/add/y:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2*
(control_network/layer_norm/batchnorm/addÁ
*control_network/layer_norm/batchnorm/RsqrtRsqrt,control_network/layer_norm/batchnorm/add:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2,
*control_network/layer_norm/batchnorm/Rsqrtđ
7control_network/layer_norm/batchnorm/mul/ReadVariableOpReadVariableOp@control_network_layer_norm_batchnorm_mul_readvariableop_resource*
_output_shapes	
:Ź*
dtype029
7control_network/layer_norm/batchnorm/mul/ReadVariableOp˙
(control_network/layer_norm/batchnorm/mulMul.control_network/layer_norm/batchnorm/Rsqrt:y:0?control_network/layer_norm/batchnorm/mul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź2*
(control_network/layer_norm/batchnorm/mulă
*control_network/layer_norm/batchnorm/mul_1Mul!control_network/normalize/Add:z:0,control_network/layer_norm/batchnorm/mul:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź2,
*control_network/layer_norm/batchnorm/mul_1ň
*control_network/layer_norm/batchnorm/mul_2Mul0control_network/layer_norm/moments/mean:output:0,control_network/layer_norm/batchnorm/mul:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź2,
*control_network/layer_norm/batchnorm/mul_2˙
<control_network/layer_norm/batchnorm/Identity/ReadVariableOpReadVariableOpEcontrol_network_layer_norm_batchnorm_identity_readvariableop_resource*
_output_shapes	
:Ź*
dtype02>
<control_network/layer_norm/batchnorm/Identity/ReadVariableOpÖ
-control_network/layer_norm/batchnorm/IdentityIdentityDcontrol_network/layer_norm/batchnorm/Identity/ReadVariableOp:value:0*
T0*
_output_shapes	
:Ź2/
-control_network/layer_norm/batchnorm/Identityö
(control_network/layer_norm/batchnorm/subSub6control_network/layer_norm/batchnorm/Identity:output:0.control_network/layer_norm/batchnorm/mul_2:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź2*
(control_network/layer_norm/batchnorm/subň
*control_network/layer_norm/batchnorm/add_1AddV2.control_network/layer_norm/batchnorm/mul_1:z:0,control_network/layer_norm/batchnorm/sub:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź2,
*control_network/layer_norm/batchnorm/add_1
control_network/TanhTanh.control_network/layer_norm/batchnorm/add_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź2
control_network/Tanh
#control_network/concat_1/concat_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#control_network/concat_1/concat_dim
control_network/concat_1/concatIdentitycontrol_network/Tanh:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź2!
control_network/concat_1/concatţ
:LayerNormAndResidualMLP/mlp/linear_0/MatMul/ReadVariableOpReadVariableOpClayernormandresidualmlp_mlp_linear_0_matmul_readvariableop_resource* 
_output_shapes
:
Ź*
dtype02<
:LayerNormAndResidualMLP/mlp/linear_0/MatMul/ReadVariableOp
+LayerNormAndResidualMLP/mlp/linear_0/MatMulMatMul(control_network/concat_1/concat:output:0BLayerNormAndResidualMLP/mlp/linear_0/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2-
+LayerNormAndResidualMLP/mlp/linear_0/MatMulđ
7LayerNormAndResidualMLP/mlp/linear_0/Add/ReadVariableOpReadVariableOp@layernormandresidualmlp_mlp_linear_0_add_readvariableop_resource*
_output_shapes	
:*
dtype029
7LayerNormAndResidualMLP/mlp/linear_0/Add/ReadVariableOp
(LayerNormAndResidualMLP/mlp/linear_0/AddAdd5LayerNormAndResidualMLP/mlp/linear_0/MatMul:product:0?LayerNormAndResidualMLP/mlp/linear_0/Add/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2*
(LayerNormAndResidualMLP/mlp/linear_0/Add
<LayerNormAndResidualMLP/mlp/linear_0/MatMul_1/ReadVariableOpReadVariableOpElayernormandresidualmlp_mlp_linear_0_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype02>
<LayerNormAndResidualMLP/mlp/linear_0/MatMul_1/ReadVariableOp
-LayerNormAndResidualMLP/mlp/linear_0/MatMul_1MatMul,LayerNormAndResidualMLP/mlp/linear_0/Add:z:0DLayerNormAndResidualMLP/mlp/linear_0/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2/
-LayerNormAndResidualMLP/mlp/linear_0/MatMul_1ö
9LayerNormAndResidualMLP/mlp/linear_0/Add_1/ReadVariableOpReadVariableOpBlayernormandresidualmlp_mlp_linear_0_add_1_readvariableop_resource*
_output_shapes	
:*
dtype02;
9LayerNormAndResidualMLP/mlp/linear_0/Add_1/ReadVariableOp
*LayerNormAndResidualMLP/mlp/linear_0/Add_1Add7LayerNormAndResidualMLP/mlp/linear_0/MatMul_1:product:0ALayerNormAndResidualMLP/mlp/linear_0/Add_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2,
*LayerNormAndResidualMLP/mlp/linear_0/Add_1Ż
 LayerNormAndResidualMLP/mlp/ReluRelu.LayerNormAndResidualMLP/mlp/linear_0/Add_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2"
 LayerNormAndResidualMLP/mlp/Reluţ
:LayerNormAndResidualMLP/mlp/linear_1/MatMul/ReadVariableOpReadVariableOpClayernormandresidualmlp_mlp_linear_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02<
:LayerNormAndResidualMLP/mlp/linear_1/MatMul/ReadVariableOp
+LayerNormAndResidualMLP/mlp/linear_1/MatMulMatMul.LayerNormAndResidualMLP/mlp/Relu:activations:0BLayerNormAndResidualMLP/mlp/linear_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2-
+LayerNormAndResidualMLP/mlp/linear_1/MatMulđ
7LayerNormAndResidualMLP/mlp/linear_1/Add/ReadVariableOpReadVariableOp@layernormandresidualmlp_mlp_linear_1_add_readvariableop_resource*
_output_shapes	
:*
dtype029
7LayerNormAndResidualMLP/mlp/linear_1/Add/ReadVariableOp
(LayerNormAndResidualMLP/mlp/linear_1/AddAdd5LayerNormAndResidualMLP/mlp/linear_1/MatMul:product:0?LayerNormAndResidualMLP/mlp/linear_1/Add/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2*
(LayerNormAndResidualMLP/mlp/linear_1/Add
4LayerNormAndResidualMLP/ResidualLayernormWrapper/addAddV2,LayerNormAndResidualMLP/mlp/linear_1/Add:z:0,LayerNormAndResidualMLP/mlp/linear_0/Add:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙26
4LayerNormAndResidualMLP/ResidualLayernormWrapper/add
ZLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙2\
ZLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments/mean/reduction_indicesţ
HLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments/meanMean8LayerNormAndResidualMLP/ResidualLayernormWrapper/add:z:0cLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments/mean/reduction_indices:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	keep_dims(2J
HLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments/meanš
PLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments/StopGradientStopGradientQLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments/mean:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2R
PLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments/StopGradient
ULayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments/SquaredDifferenceSquaredDifference8LayerNormAndResidualMLP/ResidualLayernormWrapper/add:z:0YLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments/StopGradient:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2W
ULayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments/SquaredDifference
^LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙2`
^LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments/variance/reduction_indicesŤ
LLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments/varianceMeanYLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments/SquaredDifference:z:0gLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments/variance/reduction_indices:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	keep_dims(2N
LLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments/varianceß
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŹĹ'72M
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/add/yţ
ILayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/addAddV2ULayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments/variance:output:0TLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/add/y:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2K
ILayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/add¤
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/RsqrtRsqrtMLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/add:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2M
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/RsqrtÓ
XLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/mul/ReadVariableOpReadVariableOpalayernormandresidualmlp_residuallayernormwrapper_layer_norm_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02Z
XLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/mul/ReadVariableOp
ILayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/mulMulOLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/Rsqrt:y:0`LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/mul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2K
ILayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/mulÝ
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/mul_1Mul8LayerNormAndResidualMLP/ResidualLayernormWrapper/add:z:0MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/mul:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2M
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/mul_1ö
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/mul_2MulQLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments/mean:output:0MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/mul:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2M
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/mul_2â
]LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/Identity/ReadVariableOpReadVariableOpflayernormandresidualmlp_residuallayernormwrapper_layer_norm_batchnorm_identity_readvariableop_resource*
_output_shapes	
:*
dtype02_
]LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/Identity/ReadVariableOpš
NLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/IdentityIdentityeLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/Identity/ReadVariableOp:value:0*
T0*
_output_shapes	
:2P
NLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/Identityú
ILayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/subSubWLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/Identity:output:0OLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/mul_2:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2K
ILayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/subö
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/add_1AddV2OLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/mul_1:z:0MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/sub:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2M
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/add_1
<LayerNormAndResidualMLP/mlp/linear_0/MatMul_2/ReadVariableOpReadVariableOpElayernormandresidualmlp_mlp_linear_0_matmul_2_readvariableop_resource* 
_output_shapes
:
*
dtype02>
<LayerNormAndResidualMLP/mlp/linear_0/MatMul_2/ReadVariableOp˛
-LayerNormAndResidualMLP/mlp/linear_0/MatMul_2MatMulOLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/add_1:z:0DLayerNormAndResidualMLP/mlp/linear_0/MatMul_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2/
-LayerNormAndResidualMLP/mlp/linear_0/MatMul_2ö
9LayerNormAndResidualMLP/mlp/linear_0/Add_2/ReadVariableOpReadVariableOpBlayernormandresidualmlp_mlp_linear_0_add_2_readvariableop_resource*
_output_shapes	
:*
dtype02;
9LayerNormAndResidualMLP/mlp/linear_0/Add_2/ReadVariableOp
*LayerNormAndResidualMLP/mlp/linear_0/Add_2Add7LayerNormAndResidualMLP/mlp/linear_0/MatMul_2:product:0ALayerNormAndResidualMLP/mlp/linear_0/Add_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2,
*LayerNormAndResidualMLP/mlp/linear_0/Add_2ł
"LayerNormAndResidualMLP/mlp/Relu_1Relu.LayerNormAndResidualMLP/mlp/linear_0/Add_2:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2$
"LayerNormAndResidualMLP/mlp/Relu_1
<LayerNormAndResidualMLP/mlp/linear_1/MatMul_1/ReadVariableOpReadVariableOpElayernormandresidualmlp_mlp_linear_1_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype02>
<LayerNormAndResidualMLP/mlp/linear_1/MatMul_1/ReadVariableOp
-LayerNormAndResidualMLP/mlp/linear_1/MatMul_1MatMul0LayerNormAndResidualMLP/mlp/Relu_1:activations:0DLayerNormAndResidualMLP/mlp/linear_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2/
-LayerNormAndResidualMLP/mlp/linear_1/MatMul_1ö
9LayerNormAndResidualMLP/mlp/linear_1/Add_1/ReadVariableOpReadVariableOpBlayernormandresidualmlp_mlp_linear_1_add_1_readvariableop_resource*
_output_shapes	
:*
dtype02;
9LayerNormAndResidualMLP/mlp/linear_1/Add_1/ReadVariableOp
*LayerNormAndResidualMLP/mlp/linear_1/Add_1Add7LayerNormAndResidualMLP/mlp/linear_1/MatMul_1:product:0ALayerNormAndResidualMLP/mlp/linear_1/Add_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2,
*LayerNormAndResidualMLP/mlp/linear_1/Add_1­
6LayerNormAndResidualMLP/ResidualLayernormWrapper/add_1AddV2.LayerNormAndResidualMLP/mlp/linear_1/Add_1:z:0OLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/add_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙28
6LayerNormAndResidualMLP/ResidualLayernormWrapper/add_1
\LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙2^
\LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_1/mean/reduction_indices
JLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_1/meanMean:LayerNormAndResidualMLP/ResidualLayernormWrapper/add_1:z:0eLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_1/mean/reduction_indices:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	keep_dims(2L
JLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_1/meanż
RLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_1/StopGradientStopGradientSLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_1/mean:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2T
RLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_1/StopGradient
WLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_1/SquaredDifferenceSquaredDifference:LayerNormAndResidualMLP/ResidualLayernormWrapper/add_1:z:0[LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_1/StopGradient:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2Y
WLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_1/SquaredDifference
`LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙2b
`LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_1/variance/reduction_indicesł
NLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_1/varianceMean[LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_1/SquaredDifference:z:0iLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_1/variance/reduction_indices:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	keep_dims(2P
NLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_1/varianceă
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŹĹ'72O
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/add/y
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/addAddV2WLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_1/variance:output:0VLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/add/y:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2M
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/addŞ
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/RsqrtRsqrtOLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/add:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2O
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/RsqrtŮ
ZLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/mul/ReadVariableOpReadVariableOpclayernormandresidualmlp_residuallayernormwrapper_layer_norm_batchnorm_1_mul_readvariableop_resource*
_output_shapes	
:*
dtype02\
ZLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/mul/ReadVariableOp
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/mulMulQLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/Rsqrt:y:0bLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/mul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2M
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/mulĺ
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/mul_1Mul:LayerNormAndResidualMLP/ResidualLayernormWrapper/add_1:z:0OLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/mul:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2O
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/mul_1ţ
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/mul_2MulSLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_1/mean:output:0OLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/mul:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2O
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/mul_2č
_LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/Identity/ReadVariableOpReadVariableOphlayernormandresidualmlp_residuallayernormwrapper_layer_norm_batchnorm_1_identity_readvariableop_resource*
_output_shapes	
:*
dtype02a
_LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/Identity/ReadVariableOpż
PLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/IdentityIdentitygLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/Identity/ReadVariableOp:value:0*
T0*
_output_shapes	
:2R
PLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/Identity
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/subSubYLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/Identity:output:0QLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/mul_2:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2M
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/subţ
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/add_1AddV2QLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/mul_1:z:0OLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/sub:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2O
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/add_1
<LayerNormAndResidualMLP/mlp/linear_0/MatMul_3/ReadVariableOpReadVariableOpElayernormandresidualmlp_mlp_linear_0_matmul_3_readvariableop_resource* 
_output_shapes
:
*
dtype02>
<LayerNormAndResidualMLP/mlp/linear_0/MatMul_3/ReadVariableOp´
-LayerNormAndResidualMLP/mlp/linear_0/MatMul_3MatMulQLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/add_1:z:0DLayerNormAndResidualMLP/mlp/linear_0/MatMul_3/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2/
-LayerNormAndResidualMLP/mlp/linear_0/MatMul_3ö
9LayerNormAndResidualMLP/mlp/linear_0/Add_3/ReadVariableOpReadVariableOpBlayernormandresidualmlp_mlp_linear_0_add_3_readvariableop_resource*
_output_shapes	
:*
dtype02;
9LayerNormAndResidualMLP/mlp/linear_0/Add_3/ReadVariableOp
*LayerNormAndResidualMLP/mlp/linear_0/Add_3Add7LayerNormAndResidualMLP/mlp/linear_0/MatMul_3:product:0ALayerNormAndResidualMLP/mlp/linear_0/Add_3/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2,
*LayerNormAndResidualMLP/mlp/linear_0/Add_3ł
"LayerNormAndResidualMLP/mlp/Relu_2Relu.LayerNormAndResidualMLP/mlp/linear_0/Add_3:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2$
"LayerNormAndResidualMLP/mlp/Relu_2
<LayerNormAndResidualMLP/mlp/linear_1/MatMul_2/ReadVariableOpReadVariableOpElayernormandresidualmlp_mlp_linear_1_matmul_2_readvariableop_resource* 
_output_shapes
:
*
dtype02>
<LayerNormAndResidualMLP/mlp/linear_1/MatMul_2/ReadVariableOp
-LayerNormAndResidualMLP/mlp/linear_1/MatMul_2MatMul0LayerNormAndResidualMLP/mlp/Relu_2:activations:0DLayerNormAndResidualMLP/mlp/linear_1/MatMul_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2/
-LayerNormAndResidualMLP/mlp/linear_1/MatMul_2ö
9LayerNormAndResidualMLP/mlp/linear_1/Add_2/ReadVariableOpReadVariableOpBlayernormandresidualmlp_mlp_linear_1_add_2_readvariableop_resource*
_output_shapes	
:*
dtype02;
9LayerNormAndResidualMLP/mlp/linear_1/Add_2/ReadVariableOp
*LayerNormAndResidualMLP/mlp/linear_1/Add_2Add7LayerNormAndResidualMLP/mlp/linear_1/MatMul_2:product:0ALayerNormAndResidualMLP/mlp/linear_1/Add_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2,
*LayerNormAndResidualMLP/mlp/linear_1/Add_2Ż
6LayerNormAndResidualMLP/ResidualLayernormWrapper/add_2AddV2.LayerNormAndResidualMLP/mlp/linear_1/Add_2:z:0QLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/add_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙28
6LayerNormAndResidualMLP/ResidualLayernormWrapper/add_2
\LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_2/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙2^
\LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_2/mean/reduction_indices
JLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_2/meanMean:LayerNormAndResidualMLP/ResidualLayernormWrapper/add_2:z:0eLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_2/mean/reduction_indices:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	keep_dims(2L
JLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_2/meanż
RLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_2/StopGradientStopGradientSLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_2/mean:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2T
RLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_2/StopGradient
WLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_2/SquaredDifferenceSquaredDifference:LayerNormAndResidualMLP/ResidualLayernormWrapper/add_2:z:0[LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_2/StopGradient:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2Y
WLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_2/SquaredDifference
`LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_2/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙2b
`LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_2/variance/reduction_indicesł
NLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_2/varianceMean[LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_2/SquaredDifference:z:0iLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_2/variance/reduction_indices:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	keep_dims(2P
NLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_2/varianceă
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŹĹ'72O
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/add/y
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/addAddV2WLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_2/variance:output:0VLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/add/y:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2M
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/addŞ
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/RsqrtRsqrtOLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/add:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2O
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/RsqrtŮ
ZLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/mul/ReadVariableOpReadVariableOpclayernormandresidualmlp_residuallayernormwrapper_layer_norm_batchnorm_2_mul_readvariableop_resource*
_output_shapes	
:*
dtype02\
ZLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/mul/ReadVariableOp
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/mulMulQLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/Rsqrt:y:0bLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/mul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2M
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/mulĺ
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/mul_1Mul:LayerNormAndResidualMLP/ResidualLayernormWrapper/add_2:z:0OLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/mul:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2O
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/mul_1ţ
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/mul_2MulSLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_2/mean:output:0OLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/mul:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2O
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/mul_2č
_LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/Identity/ReadVariableOpReadVariableOphlayernormandresidualmlp_residuallayernormwrapper_layer_norm_batchnorm_2_identity_readvariableop_resource*
_output_shapes	
:*
dtype02a
_LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/Identity/ReadVariableOpż
PLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/IdentityIdentitygLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/Identity/ReadVariableOp:value:0*
T0*
_output_shapes	
:2R
PLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/Identity
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/subSubYLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/Identity:output:0QLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/mul_2:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2M
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/subţ
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/add_1AddV2QLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/mul_1:z:0OLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/sub:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2O
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/add_1
<LayerNormAndResidualMLP/mlp/linear_0/MatMul_4/ReadVariableOpReadVariableOpElayernormandresidualmlp_mlp_linear_0_matmul_4_readvariableop_resource* 
_output_shapes
:
*
dtype02>
<LayerNormAndResidualMLP/mlp/linear_0/MatMul_4/ReadVariableOp´
-LayerNormAndResidualMLP/mlp/linear_0/MatMul_4MatMulQLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/add_1:z:0DLayerNormAndResidualMLP/mlp/linear_0/MatMul_4/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2/
-LayerNormAndResidualMLP/mlp/linear_0/MatMul_4ö
9LayerNormAndResidualMLP/mlp/linear_0/Add_4/ReadVariableOpReadVariableOpBlayernormandresidualmlp_mlp_linear_0_add_4_readvariableop_resource*
_output_shapes	
:*
dtype02;
9LayerNormAndResidualMLP/mlp/linear_0/Add_4/ReadVariableOp
*LayerNormAndResidualMLP/mlp/linear_0/Add_4Add7LayerNormAndResidualMLP/mlp/linear_0/MatMul_4:product:0ALayerNormAndResidualMLP/mlp/linear_0/Add_4/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2,
*LayerNormAndResidualMLP/mlp/linear_0/Add_4ł
"LayerNormAndResidualMLP/mlp/Relu_3Relu.LayerNormAndResidualMLP/mlp/linear_0/Add_4:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2$
"LayerNormAndResidualMLP/mlp/Relu_3
<LayerNormAndResidualMLP/mlp/linear_1/MatMul_3/ReadVariableOpReadVariableOpElayernormandresidualmlp_mlp_linear_1_matmul_3_readvariableop_resource* 
_output_shapes
:
*
dtype02>
<LayerNormAndResidualMLP/mlp/linear_1/MatMul_3/ReadVariableOp
-LayerNormAndResidualMLP/mlp/linear_1/MatMul_3MatMul0LayerNormAndResidualMLP/mlp/Relu_3:activations:0DLayerNormAndResidualMLP/mlp/linear_1/MatMul_3/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2/
-LayerNormAndResidualMLP/mlp/linear_1/MatMul_3ö
9LayerNormAndResidualMLP/mlp/linear_1/Add_3/ReadVariableOpReadVariableOpBlayernormandresidualmlp_mlp_linear_1_add_3_readvariableop_resource*
_output_shapes	
:*
dtype02;
9LayerNormAndResidualMLP/mlp/linear_1/Add_3/ReadVariableOp
*LayerNormAndResidualMLP/mlp/linear_1/Add_3Add7LayerNormAndResidualMLP/mlp/linear_1/MatMul_3:product:0ALayerNormAndResidualMLP/mlp/linear_1/Add_3/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2,
*LayerNormAndResidualMLP/mlp/linear_1/Add_3Ż
6LayerNormAndResidualMLP/ResidualLayernormWrapper/add_3AddV2.LayerNormAndResidualMLP/mlp/linear_1/Add_3:z:0QLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/add_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙28
6LayerNormAndResidualMLP/ResidualLayernormWrapper/add_3
\LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_3/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙2^
\LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_3/mean/reduction_indices
JLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_3/meanMean:LayerNormAndResidualMLP/ResidualLayernormWrapper/add_3:z:0eLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_3/mean/reduction_indices:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	keep_dims(2L
JLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_3/meanż
RLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_3/StopGradientStopGradientSLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_3/mean:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2T
RLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_3/StopGradient
WLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_3/SquaredDifferenceSquaredDifference:LayerNormAndResidualMLP/ResidualLayernormWrapper/add_3:z:0[LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_3/StopGradient:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2Y
WLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_3/SquaredDifference
`LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_3/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙2b
`LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_3/variance/reduction_indicesł
NLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_3/varianceMean[LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_3/SquaredDifference:z:0iLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_3/variance/reduction_indices:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	keep_dims(2P
NLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_3/varianceă
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŹĹ'72O
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/add/y
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/addAddV2WLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_3/variance:output:0VLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/add/y:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2M
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/addŞ
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/RsqrtRsqrtOLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/add:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2O
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/RsqrtŮ
ZLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/mul/ReadVariableOpReadVariableOpclayernormandresidualmlp_residuallayernormwrapper_layer_norm_batchnorm_3_mul_readvariableop_resource*
_output_shapes	
:*
dtype02\
ZLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/mul/ReadVariableOp
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/mulMulQLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/Rsqrt:y:0bLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/mul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2M
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/mulĺ
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/mul_1Mul:LayerNormAndResidualMLP/ResidualLayernormWrapper/add_3:z:0OLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/mul:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2O
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/mul_1ţ
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/mul_2MulSLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_3/mean:output:0OLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/mul:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2O
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/mul_2č
_LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/Identity/ReadVariableOpReadVariableOphlayernormandresidualmlp_residuallayernormwrapper_layer_norm_batchnorm_3_identity_readvariableop_resource*
_output_shapes	
:*
dtype02a
_LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/Identity/ReadVariableOpż
PLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/IdentityIdentitygLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/Identity/ReadVariableOp:value:0*
T0*
_output_shapes	
:2R
PLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/Identity
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/subSubYLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/Identity:output:0QLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/mul_2:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2M
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/subţ
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/add_1AddV2QLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/mul_1:z:0OLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/sub:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2O
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/add_1
 MultivariateNormalDiagHead/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 MultivariateNormalDiagHead/Constô
7MultivariateNormalDiagHead/linear/MatMul/ReadVariableOpReadVariableOp@multivariatenormaldiaghead_linear_matmul_readvariableop_resource*
_output_shapes
:	*
dtype029
7MultivariateNormalDiagHead/linear/MatMul/ReadVariableOp¤
(MultivariateNormalDiagHead/linear/MatMulMatMulQLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/add_1:z:0?MultivariateNormalDiagHead/linear/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2*
(MultivariateNormalDiagHead/linear/MatMulć
4MultivariateNormalDiagHead/linear/Add/ReadVariableOpReadVariableOp=multivariatenormaldiaghead_linear_add_readvariableop_resource*
_output_shapes
:*
dtype026
4MultivariateNormalDiagHead/linear/Add/ReadVariableOpů
%MultivariateNormalDiagHead/linear/AddAdd2MultivariateNormalDiagHead/linear/MatMul:product:0<MultivariateNormalDiagHead/linear/Add/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2'
%MultivariateNormalDiagHead/linear/Addú
9MultivariateNormalDiagHead/linear/MatMul_1/ReadVariableOpReadVariableOpBmultivariatenormaldiaghead_linear_matmul_1_readvariableop_resource*
_output_shapes
:	*
dtype02;
9MultivariateNormalDiagHead/linear/MatMul_1/ReadVariableOpŞ
*MultivariateNormalDiagHead/linear/MatMul_1MatMulQLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/add_1:z:0AMultivariateNormalDiagHead/linear/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2,
*MultivariateNormalDiagHead/linear/MatMul_1ě
6MultivariateNormalDiagHead/linear/Add_1/ReadVariableOpReadVariableOp?multivariatenormaldiaghead_linear_add_1_readvariableop_resource*
_output_shapes
:*
dtype028
6MultivariateNormalDiagHead/linear/Add_1/ReadVariableOp
'MultivariateNormalDiagHead/linear/Add_1Add4MultivariateNormalDiagHead/linear/MatMul_1:product:0>MultivariateNormalDiagHead/linear/Add_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2)
'MultivariateNormalDiagHead/linear/Add_1ľ
#MultivariateNormalDiagHead/SoftplusSoftplus+MultivariateNormalDiagHead/linear/Add_1:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2%
#MultivariateNormalDiagHead/SoftplusŚ
%MultivariateNormalDiagHead/Softplus_1Softplus)MultivariateNormalDiagHead/Const:output:0*
T0*
_output_shapes
: 2'
%MultivariateNormalDiagHead/Softplus_1
$MultivariateNormalDiagHead/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *>2&
$MultivariateNormalDiagHead/truediv/xŘ
"MultivariateNormalDiagHead/truedivRealDiv-MultivariateNormalDiagHead/truediv/x:output:03MultivariateNormalDiagHead/Softplus_1:activations:0*
T0*
_output_shapes
: 2$
"MultivariateNormalDiagHead/truedivÔ
MultivariateNormalDiagHead/mulMul1MultivariateNormalDiagHead/Softplus:activations:0&MultivariateNormalDiagHead/truediv:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2 
MultivariateNormalDiagHead/mul
 MultivariateNormalDiagHead/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *˝752"
 MultivariateNormalDiagHead/add/yĘ
MultivariateNormalDiagHead/addAddV2"MultivariateNormalDiagHead/mul:z:0)MultivariateNormalDiagHead/add/y:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2 
MultivariateNormalDiagHead/addź
{MultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/range_dimension_tensor/ConstConst*
_output_shapes
: *
dtype0*
value	B :2}
{MultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/range_dimension_tensor/Constü
WMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/event_shapeConst*
_output_shapes
:*
dtype0*
valueB:2Y
WMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/event_shape
MultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/ShapeShape"MultivariateNormalDiagHead/add:z:0*
T0*
_output_shapes
:2
MultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/Shape¤
ĽMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙2¨
ĽMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack
§MultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2Ş
§MultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_1
§MultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2Ş
§MultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_2Ő
MultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_sliceStridedSlice MultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/Shape:output:0ŽMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack:output:0°MultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_1:output:0°MultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2˘
MultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_sliceŞ
ĄMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/values_1Pack¨MultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice:output:0*
N*
T0*
_output_shapes
:2¤
ĄMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/values_1
MultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
MultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/axisę
MultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concatConcatV2 MultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/Shape:output:0ŞMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/values_1:output:0ŚMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/axis:output:0*
N*
T0*
_output_shapes
:2
MultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concatŰ
MultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
MultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stackč
MultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ţ˙˙˙˙˙˙˙˙2
MultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_1ß
MultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
MultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_2˛
MultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_sliceStridedSliceĄMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat:output:0MultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack:output:0MultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_1:output:0MultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
MultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice˙
QMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/ShapeShape)MultivariateNormalDiagHead/linear/Add:z:0*
T0*
_output_shapes
:2S
QMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/Shape
_MultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2a
_MultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack
aMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙2c
aMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_1
aMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2c
aMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_2¨
YMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_sliceStridedSliceZMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/Shape:output:0hMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack:output:0jMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_1:output:0jMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2[
YMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice
wMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/prefer_static_broadcast_shape/BroadcastArgsBroadcastArgsMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice:output:0bMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice:output:0*
_output_shapes
:2y
wMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/prefer_static_broadcast_shape/BroadcastArgsĂ
=MultivariateNormalDiagHead/MultivariateNormalDiag/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2?
=MultivariateNormalDiagHead/MultivariateNormalDiag/zeros/Constî
7MultivariateNormalDiagHead/MultivariateNormalDiag/zerosFill|MultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/prefer_static_broadcast_shape/BroadcastArgs:r0:0FMultivariateNormalDiagHead/MultivariateNormalDiag/zeros/Const:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙29
7MultivariateNormalDiagHead/MultivariateNormalDiag/zerosľ
6MultivariateNormalDiagHead/MultivariateNormalDiag/onesConst*
_output_shapes
: *
dtype0*
valueB
 *  ?28
6MultivariateNormalDiagHead/MultivariateNormalDiag/ones˛
6MultivariateNormalDiagHead/MultivariateNormalDiag/zeroConst*
_output_shapes
: *
dtype0*
value	B : 28
6MultivariateNormalDiagHead/MultivariateNormalDiag/zeroľ
7MultivariateNormalDiagHead/MultivariateNormalDiag/emptyConst*
_output_shapes
: *
dtype0*
valueB 29
7MultivariateNormalDiagHead/MultivariateNormalDiag/empty
^stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB 2`
^stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/sample_shape
âstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal/is_scalar_batch/is_scalar_batchConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2ĺ
âstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal/is_scalar_batch/is_scalar_batch
bstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/pick_vector/condConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2d
bstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/pick_vector/cond˘
jstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/pick_vector/false_vectorConst*
_output_shapes
:*
dtype0*
valueB:2l
jstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/pick_vector/false_vector
dstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/pick_vector_1/condConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2f
dstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/pick_vector_1/cond¤
kstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/pick_vector_1/true_vectorConst*
_output_shapes
:*
dtype0*
valueB:2m
kstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/pick_vector_1/true_vector
Ústochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/sample_shape/xConst*
_output_shapes
:*
dtype0*
valueB:2Ý
Ústochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/sample_shape/x
stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/sample_shape/xConst*
_output_shapes
:*
dtype0*
valueB"      2
stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/sample_shape/x
stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/ShapeShape@MultivariateNormalDiagHead/MultivariateNormalDiag/zeros:output:0*
T0*
_output_shapes
:2
stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/Shape
stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/BroadcastArgs/s1Const*
_output_shapes
: *
dtype0*
valueB 2
stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/BroadcastArgs/s1Ť	
stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/BroadcastArgsBroadcastArgsstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/Shape:output:0Ľstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/BroadcastArgs/s1:output:0*
_output_shapes
:2
stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/BroadcastArgs
stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:2
stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/concat/values_0÷
stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/concat/axisĐ
stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/concatConcatV2¤stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/concat/values_0:output:0stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/BroadcastArgs:r0:0 stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2
stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/concat
Ľstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2¨
Ľstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/normal/random_normal/mean
§stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2Ş
§stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/normal/random_normal/stddevć
ľstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormalstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/concat:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype02¸
ľstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/normal/random_normal/RandomStandardNormalü	
¤stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/normal/random_normal/mulMulžstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/normal/random_normal/RandomStandardNormal:output:0°stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/normal/random_normal/stddev:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2§
¤stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/normal/random_normal/mulÜ	
 stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/normal/random_normalAdd¨stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/normal/random_normal/mul:z:0Žstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/normal/random_normal/mean:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2Ł
 stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/normal/random_normalĆ
stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/mulMul¤stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/normal/random_normal:z:0?MultivariateNormalDiagHead/MultivariateNormalDiag/ones:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/mul¸
stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/addAddV2stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/mul:z:0@MultivariateNormalDiagHead/MultivariateNormalDiag/zeros:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/addń
stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/Shape_1Shapestochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/add:z:0*
T0*
_output_shapes
:2
stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/Shape_1
stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2˘
stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/strided_slice/stack
Ąstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2¤
Ąstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/strided_slice/stack_1
Ąstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2¤
Ąstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/strided_slice/stack_2Ż
stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/strided_sliceStridedSlicestochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/Shape_1:output:0¨stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/strided_slice/stack:output:0Şstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/strided_slice/stack_1:output:0Şstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2
stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/strided_sliceű
stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/concat_1/axisŮ
stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/concat_1ConcatV2Łstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/sample_shape/x:output:0˘stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/strided_slice:output:0˘stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/concat_1¤	
stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/ReshapeReshapestochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/add:z:0stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/concat_1:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙2
stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/Reshape
Ústochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2Ý
Ústochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/transpose/permů
Őstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/transpose	Transposestochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/Reshape:output:0ăstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/transpose/perm:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙2Ř
Őstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/transposeł
Ństochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/ShapeShapeŮstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/transpose:y:0*
T0*
_output_shapes
:2Ô
Ństochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/Shape
ßstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2â
ßstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/strided_slice/stack
ástochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2ä
ástochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/strided_slice/stack_1
ástochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2ä
ástochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/strided_slice/stack_2­
Ůstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/strided_sliceStridedSliceÚstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/Shape:output:0čstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/strided_slice/stack:output:0ęstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/strided_slice/stack_1:output:0ęstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2Ü
Ůstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/strided_slice÷
×stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2Ú
×stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/concat/axis	
Ňstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/concatConcatV2ăstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/sample_shape/x:output:0âstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/strided_slice:output:0ŕstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2Ő
Ňstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/concat¨
Óstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/ReshapeReshapeŮstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/transpose:y:0Űstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/concat:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙2Ö
Óstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/Reshapeż
Wstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/ShapeShapeÜstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/Reshape:output:0*
T0*
_output_shapes
:2Y
Wstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/Shape
estochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2g
estochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/strided_slice/stack
gstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2i
gstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/strided_slice/stack_1
gstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2i
gstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/strided_slice/stack_2Ę
_stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/strided_sliceStridedSlice`stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/Shape:output:0nstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/strided_slice/stack:output:0pstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/strided_slice/stack_1:output:0pstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2a
_stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/strided_slice
]stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2_
]stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/concat/axisŠ
Xstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/concatConcatV2gstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/sample_shape:output:0hstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/strided_slice:output:0fstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2Z
Xstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/concatľ
Ystochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/ReshapeReshapeÜstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/Reshape:output:0astochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/concat:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2[
Ystochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/Reshapeî
stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_chain_of_MultivariateNormalDiagHead_MultivariateNormalDiag_shift_of_MultivariateNormalDiagHead_MultivariateNormalDiag_scale_matvec_linear_operator/forward/MultivariateNormalDiagHead_MultivariateNormalDiag_scale_matvec_linear_operator/forward/LinearOperatorDiag/matvec/mulMul"MultivariateNormalDiagHead/add:z:0bstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/Reshape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_chain_of_MultivariateNormalDiagHead_MultivariateNormalDiag_shift_of_MultivariateNormalDiagHead_MultivariateNormalDiag_scale_matvec_linear_operator/forward/MultivariateNormalDiagHead_MultivariateNormalDiag_scale_matvec_linear_operator/forward/LinearOperatorDiag/matvec/mulË	
âstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_chain_of_MultivariateNormalDiagHead_MultivariateNormalDiag_shift_of_MultivariateNormalDiagHead_MultivariateNormalDiag_scale_matvec_linear_operator/forward/MultivariateNormalDiagHead_MultivariateNormalDiag_shift/forward/addAddV2stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_chain_of_MultivariateNormalDiagHead_MultivariateNormalDiag_shift_of_MultivariateNormalDiagHead_MultivariateNormalDiag_scale_matvec_linear_operator/forward/MultivariateNormalDiagHead_MultivariateNormalDiag_scale_matvec_linear_operator/forward/LinearOperatorDiag/matvec/mul:z:0)MultivariateNormalDiagHead/linear/Add:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2ĺ
âstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_chain_of_MultivariateNormalDiagHead_MultivariateNormalDiag_shift_of_MultivariateNormalDiagHead_MultivariateNormalDiag_scale_matvec_linear_operator/forward/MultivariateNormalDiagHead_MultivariateNormalDiag_shift/forward/addť
IdentityIdentityćstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_chain_of_MultivariateNormalDiagHead_MultivariateNormalDiag_shift_of_MultivariateNormalDiagHead_MultivariateNormalDiag_scale_matvec_linear_operator/forward/MultivariateNormalDiagHead_MultivariateNormalDiag_shift/forward/add:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*Ă
_input_shapesą
Ž:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙	:::::::::::::::::::::::::::::::::::O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameargs_0:OK
'
_output_shapes
:˙˙˙˙˙˙˙˙˙	
 
_user_specified_nameargs_0:
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
: :#

_output_shapes
: 
Ú
ů
!__inference_wrapped_module_248501
args_0_position
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
identity}
control_network/flatten/ShapeShapeargs_0_position*
T0*
_output_shapes
:2
control_network/flatten/Shape¤
+control_network/flatten/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+control_network/flatten/strided_slice/stack¨
-control_network/flatten/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-control_network/flatten/strided_slice/stack_1¨
-control_network/flatten/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-control_network/flatten/strided_slice/stack_2đ
%control_network/flatten/strided_sliceStridedSlice&control_network/flatten/Shape:output:04control_network/flatten/strided_slice/stack:output:06control_network/flatten/strided_slice/stack_1:output:06control_network/flatten/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2'
%control_network/flatten/strided_slice
'control_network/flatten/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'control_network/flatten/concat/values_1
#control_network/flatten/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#control_network/flatten/concat/axis
control_network/flatten/concatConcatV2.control_network/flatten/strided_slice:output:00control_network/flatten/concat/values_1:output:0,control_network/flatten/concat/axis:output:0*
N*
T0*
_output_shapes
:2 
control_network/flatten/concatš
control_network/flatten/ReshapeReshapeargs_0_position'control_network/flatten/concat:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2!
control_network/flatten/Reshape
control_network/flatten_1/ShapeShapeargs_0_velocity*
T0*
_output_shapes
:2!
control_network/flatten_1/Shape¨
-control_network/flatten_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-control_network/flatten_1/strided_slice/stackŹ
/control_network/flatten_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/control_network/flatten_1/strided_slice/stack_1Ź
/control_network/flatten_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/control_network/flatten_1/strided_slice/stack_2ü
'control_network/flatten_1/strided_sliceStridedSlice(control_network/flatten_1/Shape:output:06control_network/flatten_1/strided_slice/stack:output:08control_network/flatten_1/strided_slice/stack_1:output:08control_network/flatten_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2)
'control_network/flatten_1/strided_slice 
)control_network/flatten_1/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:	2+
)control_network/flatten_1/concat/values_1
%control_network/flatten_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2'
%control_network/flatten_1/concat/axis
 control_network/flatten_1/concatConcatV20control_network/flatten_1/strided_slice:output:02control_network/flatten_1/concat/values_1:output:0.control_network/flatten_1/concat/axis:output:0*
N*
T0*
_output_shapes
:2"
 control_network/flatten_1/concatż
!control_network/flatten_1/ReshapeReshapeargs_0_velocity)control_network/flatten_1/concat:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙	2#
!control_network/flatten_1/Reshape|
control_network/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
control_network/concat/axisó
control_network/concatConcatV2(control_network/flatten/Reshape:output:0*control_network/flatten_1/Reshape:output:0$control_network/concat/axis:output:0*
N*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
control_network/concatÜ
/control_network/normalize/MatMul/ReadVariableOpReadVariableOp8control_network_normalize_matmul_readvariableop_resource*
_output_shapes
:	Ź*
dtype021
/control_network/normalize/MatMul/ReadVariableOpŰ
 control_network/normalize/MatMulMatMulcontrol_network/concat:output:07control_network/normalize/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź2"
 control_network/normalize/MatMulĎ
,control_network/normalize/Add/ReadVariableOpReadVariableOp5control_network_normalize_add_readvariableop_resource*
_output_shapes	
:Ź*
dtype02.
,control_network/normalize/Add/ReadVariableOpÚ
control_network/normalize/AddAdd*control_network/normalize/MatMul:product:04control_network/normalize/Add/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź2
control_network/normalize/AddŔ
9control_network/layer_norm/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2;
9control_network/layer_norm/moments/mean/reduction_indices
'control_network/layer_norm/moments/meanMean!control_network/normalize/Add:z:0Bcontrol_network/layer_norm/moments/mean/reduction_indices:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	keep_dims(2)
'control_network/layer_norm/moments/meanÖ
/control_network/layer_norm/moments/StopGradientStopGradient0control_network/layer_norm/moments/mean:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙21
/control_network/layer_norm/moments/StopGradient
4control_network/layer_norm/moments/SquaredDifferenceSquaredDifference!control_network/normalize/Add:z:08control_network/layer_norm/moments/StopGradient:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź26
4control_network/layer_norm/moments/SquaredDifferenceČ
=control_network/layer_norm/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2?
=control_network/layer_norm/moments/variance/reduction_indices§
+control_network/layer_norm/moments/varianceMean8control_network/layer_norm/moments/SquaredDifference:z:0Fcontrol_network/layer_norm/moments/variance/reduction_indices:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	keep_dims(2-
+control_network/layer_norm/moments/variance
*control_network/layer_norm/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŹĹ'72,
*control_network/layer_norm/batchnorm/add/yú
(control_network/layer_norm/batchnorm/addAddV24control_network/layer_norm/moments/variance:output:03control_network/layer_norm/batchnorm/add/y:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2*
(control_network/layer_norm/batchnorm/addÁ
*control_network/layer_norm/batchnorm/RsqrtRsqrt,control_network/layer_norm/batchnorm/add:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2,
*control_network/layer_norm/batchnorm/Rsqrtđ
7control_network/layer_norm/batchnorm/mul/ReadVariableOpReadVariableOp@control_network_layer_norm_batchnorm_mul_readvariableop_resource*
_output_shapes	
:Ź*
dtype029
7control_network/layer_norm/batchnorm/mul/ReadVariableOp˙
(control_network/layer_norm/batchnorm/mulMul.control_network/layer_norm/batchnorm/Rsqrt:y:0?control_network/layer_norm/batchnorm/mul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź2*
(control_network/layer_norm/batchnorm/mulă
*control_network/layer_norm/batchnorm/mul_1Mul!control_network/normalize/Add:z:0,control_network/layer_norm/batchnorm/mul:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź2,
*control_network/layer_norm/batchnorm/mul_1ň
*control_network/layer_norm/batchnorm/mul_2Mul0control_network/layer_norm/moments/mean:output:0,control_network/layer_norm/batchnorm/mul:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź2,
*control_network/layer_norm/batchnorm/mul_2˙
<control_network/layer_norm/batchnorm/Identity/ReadVariableOpReadVariableOpEcontrol_network_layer_norm_batchnorm_identity_readvariableop_resource*
_output_shapes	
:Ź*
dtype02>
<control_network/layer_norm/batchnorm/Identity/ReadVariableOpÖ
-control_network/layer_norm/batchnorm/IdentityIdentityDcontrol_network/layer_norm/batchnorm/Identity/ReadVariableOp:value:0*
T0*
_output_shapes	
:Ź2/
-control_network/layer_norm/batchnorm/Identityö
(control_network/layer_norm/batchnorm/subSub6control_network/layer_norm/batchnorm/Identity:output:0.control_network/layer_norm/batchnorm/mul_2:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź2*
(control_network/layer_norm/batchnorm/subň
*control_network/layer_norm/batchnorm/add_1AddV2.control_network/layer_norm/batchnorm/mul_1:z:0,control_network/layer_norm/batchnorm/sub:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź2,
*control_network/layer_norm/batchnorm/add_1
control_network/TanhTanh.control_network/layer_norm/batchnorm/add_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź2
control_network/Tanh
#control_network/concat_1/concat_dimConst*
_output_shapes
: *
dtype0*
value	B :2%
#control_network/concat_1/concat_dim
control_network/concat_1/concatIdentitycontrol_network/Tanh:y:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź2!
control_network/concat_1/concatţ
:LayerNormAndResidualMLP/mlp/linear_0/MatMul/ReadVariableOpReadVariableOpClayernormandresidualmlp_mlp_linear_0_matmul_readvariableop_resource* 
_output_shapes
:
Ź*
dtype02<
:LayerNormAndResidualMLP/mlp/linear_0/MatMul/ReadVariableOp
+LayerNormAndResidualMLP/mlp/linear_0/MatMulMatMul(control_network/concat_1/concat:output:0BLayerNormAndResidualMLP/mlp/linear_0/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2-
+LayerNormAndResidualMLP/mlp/linear_0/MatMulđ
7LayerNormAndResidualMLP/mlp/linear_0/Add/ReadVariableOpReadVariableOp@layernormandresidualmlp_mlp_linear_0_add_readvariableop_resource*
_output_shapes	
:*
dtype029
7LayerNormAndResidualMLP/mlp/linear_0/Add/ReadVariableOp
(LayerNormAndResidualMLP/mlp/linear_0/AddAdd5LayerNormAndResidualMLP/mlp/linear_0/MatMul:product:0?LayerNormAndResidualMLP/mlp/linear_0/Add/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2*
(LayerNormAndResidualMLP/mlp/linear_0/Add
<LayerNormAndResidualMLP/mlp/linear_0/MatMul_1/ReadVariableOpReadVariableOpElayernormandresidualmlp_mlp_linear_0_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype02>
<LayerNormAndResidualMLP/mlp/linear_0/MatMul_1/ReadVariableOp
-LayerNormAndResidualMLP/mlp/linear_0/MatMul_1MatMul,LayerNormAndResidualMLP/mlp/linear_0/Add:z:0DLayerNormAndResidualMLP/mlp/linear_0/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2/
-LayerNormAndResidualMLP/mlp/linear_0/MatMul_1ö
9LayerNormAndResidualMLP/mlp/linear_0/Add_1/ReadVariableOpReadVariableOpBlayernormandresidualmlp_mlp_linear_0_add_1_readvariableop_resource*
_output_shapes	
:*
dtype02;
9LayerNormAndResidualMLP/mlp/linear_0/Add_1/ReadVariableOp
*LayerNormAndResidualMLP/mlp/linear_0/Add_1Add7LayerNormAndResidualMLP/mlp/linear_0/MatMul_1:product:0ALayerNormAndResidualMLP/mlp/linear_0/Add_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2,
*LayerNormAndResidualMLP/mlp/linear_0/Add_1Ż
 LayerNormAndResidualMLP/mlp/ReluRelu.LayerNormAndResidualMLP/mlp/linear_0/Add_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2"
 LayerNormAndResidualMLP/mlp/Reluţ
:LayerNormAndResidualMLP/mlp/linear_1/MatMul/ReadVariableOpReadVariableOpClayernormandresidualmlp_mlp_linear_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02<
:LayerNormAndResidualMLP/mlp/linear_1/MatMul/ReadVariableOp
+LayerNormAndResidualMLP/mlp/linear_1/MatMulMatMul.LayerNormAndResidualMLP/mlp/Relu:activations:0BLayerNormAndResidualMLP/mlp/linear_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2-
+LayerNormAndResidualMLP/mlp/linear_1/MatMulđ
7LayerNormAndResidualMLP/mlp/linear_1/Add/ReadVariableOpReadVariableOp@layernormandresidualmlp_mlp_linear_1_add_readvariableop_resource*
_output_shapes	
:*
dtype029
7LayerNormAndResidualMLP/mlp/linear_1/Add/ReadVariableOp
(LayerNormAndResidualMLP/mlp/linear_1/AddAdd5LayerNormAndResidualMLP/mlp/linear_1/MatMul:product:0?LayerNormAndResidualMLP/mlp/linear_1/Add/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2*
(LayerNormAndResidualMLP/mlp/linear_1/Add
4LayerNormAndResidualMLP/ResidualLayernormWrapper/addAddV2,LayerNormAndResidualMLP/mlp/linear_1/Add:z:0,LayerNormAndResidualMLP/mlp/linear_0/Add:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙26
4LayerNormAndResidualMLP/ResidualLayernormWrapper/add
ZLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙2\
ZLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments/mean/reduction_indicesţ
HLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments/meanMean8LayerNormAndResidualMLP/ResidualLayernormWrapper/add:z:0cLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments/mean/reduction_indices:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	keep_dims(2J
HLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments/meanš
PLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments/StopGradientStopGradientQLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments/mean:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2R
PLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments/StopGradient
ULayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments/SquaredDifferenceSquaredDifference8LayerNormAndResidualMLP/ResidualLayernormWrapper/add:z:0YLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments/StopGradient:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2W
ULayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments/SquaredDifference
^LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙2`
^LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments/variance/reduction_indicesŤ
LLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments/varianceMeanYLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments/SquaredDifference:z:0gLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments/variance/reduction_indices:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	keep_dims(2N
LLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments/varianceß
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŹĹ'72M
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/add/yţ
ILayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/addAddV2ULayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments/variance:output:0TLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/add/y:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2K
ILayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/add¤
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/RsqrtRsqrtMLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/add:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2M
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/RsqrtÓ
XLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/mul/ReadVariableOpReadVariableOpalayernormandresidualmlp_residuallayernormwrapper_layer_norm_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02Z
XLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/mul/ReadVariableOp
ILayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/mulMulOLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/Rsqrt:y:0`LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/mul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2K
ILayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/mulÝ
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/mul_1Mul8LayerNormAndResidualMLP/ResidualLayernormWrapper/add:z:0MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/mul:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2M
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/mul_1ö
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/mul_2MulQLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments/mean:output:0MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/mul:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2M
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/mul_2â
]LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/Identity/ReadVariableOpReadVariableOpflayernormandresidualmlp_residuallayernormwrapper_layer_norm_batchnorm_identity_readvariableop_resource*
_output_shapes	
:*
dtype02_
]LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/Identity/ReadVariableOpš
NLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/IdentityIdentityeLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/Identity/ReadVariableOp:value:0*
T0*
_output_shapes	
:2P
NLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/Identityú
ILayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/subSubWLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/Identity:output:0OLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/mul_2:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2K
ILayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/subö
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/add_1AddV2OLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/mul_1:z:0MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/sub:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2M
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/add_1
<LayerNormAndResidualMLP/mlp/linear_0/MatMul_2/ReadVariableOpReadVariableOpElayernormandresidualmlp_mlp_linear_0_matmul_2_readvariableop_resource* 
_output_shapes
:
*
dtype02>
<LayerNormAndResidualMLP/mlp/linear_0/MatMul_2/ReadVariableOp˛
-LayerNormAndResidualMLP/mlp/linear_0/MatMul_2MatMulOLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/add_1:z:0DLayerNormAndResidualMLP/mlp/linear_0/MatMul_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2/
-LayerNormAndResidualMLP/mlp/linear_0/MatMul_2ö
9LayerNormAndResidualMLP/mlp/linear_0/Add_2/ReadVariableOpReadVariableOpBlayernormandresidualmlp_mlp_linear_0_add_2_readvariableop_resource*
_output_shapes	
:*
dtype02;
9LayerNormAndResidualMLP/mlp/linear_0/Add_2/ReadVariableOp
*LayerNormAndResidualMLP/mlp/linear_0/Add_2Add7LayerNormAndResidualMLP/mlp/linear_0/MatMul_2:product:0ALayerNormAndResidualMLP/mlp/linear_0/Add_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2,
*LayerNormAndResidualMLP/mlp/linear_0/Add_2ł
"LayerNormAndResidualMLP/mlp/Relu_1Relu.LayerNormAndResidualMLP/mlp/linear_0/Add_2:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2$
"LayerNormAndResidualMLP/mlp/Relu_1
<LayerNormAndResidualMLP/mlp/linear_1/MatMul_1/ReadVariableOpReadVariableOpElayernormandresidualmlp_mlp_linear_1_matmul_1_readvariableop_resource* 
_output_shapes
:
*
dtype02>
<LayerNormAndResidualMLP/mlp/linear_1/MatMul_1/ReadVariableOp
-LayerNormAndResidualMLP/mlp/linear_1/MatMul_1MatMul0LayerNormAndResidualMLP/mlp/Relu_1:activations:0DLayerNormAndResidualMLP/mlp/linear_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2/
-LayerNormAndResidualMLP/mlp/linear_1/MatMul_1ö
9LayerNormAndResidualMLP/mlp/linear_1/Add_1/ReadVariableOpReadVariableOpBlayernormandresidualmlp_mlp_linear_1_add_1_readvariableop_resource*
_output_shapes	
:*
dtype02;
9LayerNormAndResidualMLP/mlp/linear_1/Add_1/ReadVariableOp
*LayerNormAndResidualMLP/mlp/linear_1/Add_1Add7LayerNormAndResidualMLP/mlp/linear_1/MatMul_1:product:0ALayerNormAndResidualMLP/mlp/linear_1/Add_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2,
*LayerNormAndResidualMLP/mlp/linear_1/Add_1­
6LayerNormAndResidualMLP/ResidualLayernormWrapper/add_1AddV2.LayerNormAndResidualMLP/mlp/linear_1/Add_1:z:0OLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm/add_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙28
6LayerNormAndResidualMLP/ResidualLayernormWrapper/add_1
\LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_1/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙2^
\LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_1/mean/reduction_indices
JLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_1/meanMean:LayerNormAndResidualMLP/ResidualLayernormWrapper/add_1:z:0eLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_1/mean/reduction_indices:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	keep_dims(2L
JLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_1/meanż
RLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_1/StopGradientStopGradientSLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_1/mean:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2T
RLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_1/StopGradient
WLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_1/SquaredDifferenceSquaredDifference:LayerNormAndResidualMLP/ResidualLayernormWrapper/add_1:z:0[LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_1/StopGradient:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2Y
WLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_1/SquaredDifference
`LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_1/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙2b
`LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_1/variance/reduction_indicesł
NLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_1/varianceMean[LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_1/SquaredDifference:z:0iLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_1/variance/reduction_indices:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	keep_dims(2P
NLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_1/varianceă
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŹĹ'72O
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/add/y
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/addAddV2WLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_1/variance:output:0VLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/add/y:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2M
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/addŞ
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/RsqrtRsqrtOLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/add:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2O
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/RsqrtŮ
ZLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/mul/ReadVariableOpReadVariableOpclayernormandresidualmlp_residuallayernormwrapper_layer_norm_batchnorm_1_mul_readvariableop_resource*
_output_shapes	
:*
dtype02\
ZLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/mul/ReadVariableOp
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/mulMulQLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/Rsqrt:y:0bLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/mul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2M
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/mulĺ
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/mul_1Mul:LayerNormAndResidualMLP/ResidualLayernormWrapper/add_1:z:0OLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/mul:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2O
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/mul_1ţ
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/mul_2MulSLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_1/mean:output:0OLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/mul:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2O
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/mul_2č
_LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/Identity/ReadVariableOpReadVariableOphlayernormandresidualmlp_residuallayernormwrapper_layer_norm_batchnorm_1_identity_readvariableop_resource*
_output_shapes	
:*
dtype02a
_LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/Identity/ReadVariableOpż
PLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/IdentityIdentitygLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/Identity/ReadVariableOp:value:0*
T0*
_output_shapes	
:2R
PLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/Identity
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/subSubYLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/Identity:output:0QLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/mul_2:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2M
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/subţ
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/add_1AddV2QLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/mul_1:z:0OLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/sub:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2O
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/add_1
<LayerNormAndResidualMLP/mlp/linear_0/MatMul_3/ReadVariableOpReadVariableOpElayernormandresidualmlp_mlp_linear_0_matmul_3_readvariableop_resource* 
_output_shapes
:
*
dtype02>
<LayerNormAndResidualMLP/mlp/linear_0/MatMul_3/ReadVariableOp´
-LayerNormAndResidualMLP/mlp/linear_0/MatMul_3MatMulQLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/add_1:z:0DLayerNormAndResidualMLP/mlp/linear_0/MatMul_3/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2/
-LayerNormAndResidualMLP/mlp/linear_0/MatMul_3ö
9LayerNormAndResidualMLP/mlp/linear_0/Add_3/ReadVariableOpReadVariableOpBlayernormandresidualmlp_mlp_linear_0_add_3_readvariableop_resource*
_output_shapes	
:*
dtype02;
9LayerNormAndResidualMLP/mlp/linear_0/Add_3/ReadVariableOp
*LayerNormAndResidualMLP/mlp/linear_0/Add_3Add7LayerNormAndResidualMLP/mlp/linear_0/MatMul_3:product:0ALayerNormAndResidualMLP/mlp/linear_0/Add_3/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2,
*LayerNormAndResidualMLP/mlp/linear_0/Add_3ł
"LayerNormAndResidualMLP/mlp/Relu_2Relu.LayerNormAndResidualMLP/mlp/linear_0/Add_3:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2$
"LayerNormAndResidualMLP/mlp/Relu_2
<LayerNormAndResidualMLP/mlp/linear_1/MatMul_2/ReadVariableOpReadVariableOpElayernormandresidualmlp_mlp_linear_1_matmul_2_readvariableop_resource* 
_output_shapes
:
*
dtype02>
<LayerNormAndResidualMLP/mlp/linear_1/MatMul_2/ReadVariableOp
-LayerNormAndResidualMLP/mlp/linear_1/MatMul_2MatMul0LayerNormAndResidualMLP/mlp/Relu_2:activations:0DLayerNormAndResidualMLP/mlp/linear_1/MatMul_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2/
-LayerNormAndResidualMLP/mlp/linear_1/MatMul_2ö
9LayerNormAndResidualMLP/mlp/linear_1/Add_2/ReadVariableOpReadVariableOpBlayernormandresidualmlp_mlp_linear_1_add_2_readvariableop_resource*
_output_shapes	
:*
dtype02;
9LayerNormAndResidualMLP/mlp/linear_1/Add_2/ReadVariableOp
*LayerNormAndResidualMLP/mlp/linear_1/Add_2Add7LayerNormAndResidualMLP/mlp/linear_1/MatMul_2:product:0ALayerNormAndResidualMLP/mlp/linear_1/Add_2/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2,
*LayerNormAndResidualMLP/mlp/linear_1/Add_2Ż
6LayerNormAndResidualMLP/ResidualLayernormWrapper/add_2AddV2.LayerNormAndResidualMLP/mlp/linear_1/Add_2:z:0QLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_1/add_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙28
6LayerNormAndResidualMLP/ResidualLayernormWrapper/add_2
\LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_2/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙2^
\LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_2/mean/reduction_indices
JLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_2/meanMean:LayerNormAndResidualMLP/ResidualLayernormWrapper/add_2:z:0eLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_2/mean/reduction_indices:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	keep_dims(2L
JLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_2/meanż
RLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_2/StopGradientStopGradientSLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_2/mean:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2T
RLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_2/StopGradient
WLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_2/SquaredDifferenceSquaredDifference:LayerNormAndResidualMLP/ResidualLayernormWrapper/add_2:z:0[LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_2/StopGradient:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2Y
WLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_2/SquaredDifference
`LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_2/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙2b
`LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_2/variance/reduction_indicesł
NLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_2/varianceMean[LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_2/SquaredDifference:z:0iLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_2/variance/reduction_indices:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	keep_dims(2P
NLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_2/varianceă
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŹĹ'72O
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/add/y
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/addAddV2WLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_2/variance:output:0VLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/add/y:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2M
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/addŞ
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/RsqrtRsqrtOLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/add:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2O
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/RsqrtŮ
ZLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/mul/ReadVariableOpReadVariableOpclayernormandresidualmlp_residuallayernormwrapper_layer_norm_batchnorm_2_mul_readvariableop_resource*
_output_shapes	
:*
dtype02\
ZLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/mul/ReadVariableOp
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/mulMulQLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/Rsqrt:y:0bLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/mul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2M
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/mulĺ
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/mul_1Mul:LayerNormAndResidualMLP/ResidualLayernormWrapper/add_2:z:0OLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/mul:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2O
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/mul_1ţ
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/mul_2MulSLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_2/mean:output:0OLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/mul:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2O
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/mul_2č
_LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/Identity/ReadVariableOpReadVariableOphlayernormandresidualmlp_residuallayernormwrapper_layer_norm_batchnorm_2_identity_readvariableop_resource*
_output_shapes	
:*
dtype02a
_LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/Identity/ReadVariableOpż
PLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/IdentityIdentitygLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/Identity/ReadVariableOp:value:0*
T0*
_output_shapes	
:2R
PLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/Identity
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/subSubYLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/Identity:output:0QLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/mul_2:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2M
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/subţ
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/add_1AddV2QLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/mul_1:z:0OLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/sub:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2O
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/add_1
<LayerNormAndResidualMLP/mlp/linear_0/MatMul_4/ReadVariableOpReadVariableOpElayernormandresidualmlp_mlp_linear_0_matmul_4_readvariableop_resource* 
_output_shapes
:
*
dtype02>
<LayerNormAndResidualMLP/mlp/linear_0/MatMul_4/ReadVariableOp´
-LayerNormAndResidualMLP/mlp/linear_0/MatMul_4MatMulQLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/add_1:z:0DLayerNormAndResidualMLP/mlp/linear_0/MatMul_4/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2/
-LayerNormAndResidualMLP/mlp/linear_0/MatMul_4ö
9LayerNormAndResidualMLP/mlp/linear_0/Add_4/ReadVariableOpReadVariableOpBlayernormandresidualmlp_mlp_linear_0_add_4_readvariableop_resource*
_output_shapes	
:*
dtype02;
9LayerNormAndResidualMLP/mlp/linear_0/Add_4/ReadVariableOp
*LayerNormAndResidualMLP/mlp/linear_0/Add_4Add7LayerNormAndResidualMLP/mlp/linear_0/MatMul_4:product:0ALayerNormAndResidualMLP/mlp/linear_0/Add_4/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2,
*LayerNormAndResidualMLP/mlp/linear_0/Add_4ł
"LayerNormAndResidualMLP/mlp/Relu_3Relu.LayerNormAndResidualMLP/mlp/linear_0/Add_4:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2$
"LayerNormAndResidualMLP/mlp/Relu_3
<LayerNormAndResidualMLP/mlp/linear_1/MatMul_3/ReadVariableOpReadVariableOpElayernormandresidualmlp_mlp_linear_1_matmul_3_readvariableop_resource* 
_output_shapes
:
*
dtype02>
<LayerNormAndResidualMLP/mlp/linear_1/MatMul_3/ReadVariableOp
-LayerNormAndResidualMLP/mlp/linear_1/MatMul_3MatMul0LayerNormAndResidualMLP/mlp/Relu_3:activations:0DLayerNormAndResidualMLP/mlp/linear_1/MatMul_3/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2/
-LayerNormAndResidualMLP/mlp/linear_1/MatMul_3ö
9LayerNormAndResidualMLP/mlp/linear_1/Add_3/ReadVariableOpReadVariableOpBlayernormandresidualmlp_mlp_linear_1_add_3_readvariableop_resource*
_output_shapes	
:*
dtype02;
9LayerNormAndResidualMLP/mlp/linear_1/Add_3/ReadVariableOp
*LayerNormAndResidualMLP/mlp/linear_1/Add_3Add7LayerNormAndResidualMLP/mlp/linear_1/MatMul_3:product:0ALayerNormAndResidualMLP/mlp/linear_1/Add_3/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2,
*LayerNormAndResidualMLP/mlp/linear_1/Add_3Ż
6LayerNormAndResidualMLP/ResidualLayernormWrapper/add_3AddV2.LayerNormAndResidualMLP/mlp/linear_1/Add_3:z:0QLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_2/add_1:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙28
6LayerNormAndResidualMLP/ResidualLayernormWrapper/add_3
\LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_3/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙2^
\LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_3/mean/reduction_indices
JLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_3/meanMean:LayerNormAndResidualMLP/ResidualLayernormWrapper/add_3:z:0eLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_3/mean/reduction_indices:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	keep_dims(2L
JLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_3/meanż
RLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_3/StopGradientStopGradientSLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_3/mean:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2T
RLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_3/StopGradient
WLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_3/SquaredDifferenceSquaredDifference:LayerNormAndResidualMLP/ResidualLayernormWrapper/add_3:z:0[LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_3/StopGradient:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2Y
WLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_3/SquaredDifference
`LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_3/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙2b
`LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_3/variance/reduction_indicesł
NLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_3/varianceMean[LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_3/SquaredDifference:z:0iLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_3/variance/reduction_indices:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	keep_dims(2P
NLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_3/varianceă
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *ŹĹ'72O
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/add/y
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/addAddV2WLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_3/variance:output:0VLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/add/y:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2M
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/addŞ
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/RsqrtRsqrtOLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/add:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2O
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/RsqrtŮ
ZLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/mul/ReadVariableOpReadVariableOpclayernormandresidualmlp_residuallayernormwrapper_layer_norm_batchnorm_3_mul_readvariableop_resource*
_output_shapes	
:*
dtype02\
ZLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/mul/ReadVariableOp
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/mulMulQLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/Rsqrt:y:0bLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/mul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2M
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/mulĺ
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/mul_1Mul:LayerNormAndResidualMLP/ResidualLayernormWrapper/add_3:z:0OLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/mul:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2O
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/mul_1ţ
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/mul_2MulSLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/moments_3/mean:output:0OLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/mul:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2O
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/mul_2č
_LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/Identity/ReadVariableOpReadVariableOphlayernormandresidualmlp_residuallayernormwrapper_layer_norm_batchnorm_3_identity_readvariableop_resource*
_output_shapes	
:*
dtype02a
_LayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/Identity/ReadVariableOpż
PLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/IdentityIdentitygLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/Identity/ReadVariableOp:value:0*
T0*
_output_shapes	
:2R
PLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/Identity
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/subSubYLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/Identity:output:0QLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/mul_2:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2M
KLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/subţ
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/add_1AddV2QLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/mul_1:z:0OLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/sub:z:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2O
MLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/add_1
 MultivariateNormalDiagHead/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 MultivariateNormalDiagHead/Constô
7MultivariateNormalDiagHead/linear/MatMul/ReadVariableOpReadVariableOp@multivariatenormaldiaghead_linear_matmul_readvariableop_resource*
_output_shapes
:	*
dtype029
7MultivariateNormalDiagHead/linear/MatMul/ReadVariableOp¤
(MultivariateNormalDiagHead/linear/MatMulMatMulQLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/add_1:z:0?MultivariateNormalDiagHead/linear/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2*
(MultivariateNormalDiagHead/linear/MatMulć
4MultivariateNormalDiagHead/linear/Add/ReadVariableOpReadVariableOp=multivariatenormaldiaghead_linear_add_readvariableop_resource*
_output_shapes
:*
dtype026
4MultivariateNormalDiagHead/linear/Add/ReadVariableOpů
%MultivariateNormalDiagHead/linear/AddAdd2MultivariateNormalDiagHead/linear/MatMul:product:0<MultivariateNormalDiagHead/linear/Add/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2'
%MultivariateNormalDiagHead/linear/Addú
9MultivariateNormalDiagHead/linear/MatMul_1/ReadVariableOpReadVariableOpBmultivariatenormaldiaghead_linear_matmul_1_readvariableop_resource*
_output_shapes
:	*
dtype02;
9MultivariateNormalDiagHead/linear/MatMul_1/ReadVariableOpŞ
*MultivariateNormalDiagHead/linear/MatMul_1MatMulQLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/batchnorm_3/add_1:z:0AMultivariateNormalDiagHead/linear/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2,
*MultivariateNormalDiagHead/linear/MatMul_1ě
6MultivariateNormalDiagHead/linear/Add_1/ReadVariableOpReadVariableOp?multivariatenormaldiaghead_linear_add_1_readvariableop_resource*
_output_shapes
:*
dtype028
6MultivariateNormalDiagHead/linear/Add_1/ReadVariableOp
'MultivariateNormalDiagHead/linear/Add_1Add4MultivariateNormalDiagHead/linear/MatMul_1:product:0>MultivariateNormalDiagHead/linear/Add_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2)
'MultivariateNormalDiagHead/linear/Add_1ľ
#MultivariateNormalDiagHead/SoftplusSoftplus+MultivariateNormalDiagHead/linear/Add_1:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2%
#MultivariateNormalDiagHead/SoftplusŚ
%MultivariateNormalDiagHead/Softplus_1Softplus)MultivariateNormalDiagHead/Const:output:0*
T0*
_output_shapes
: 2'
%MultivariateNormalDiagHead/Softplus_1
$MultivariateNormalDiagHead/truediv/xConst*
_output_shapes
: *
dtype0*
valueB
 *>2&
$MultivariateNormalDiagHead/truediv/xŘ
"MultivariateNormalDiagHead/truedivRealDiv-MultivariateNormalDiagHead/truediv/x:output:03MultivariateNormalDiagHead/Softplus_1:activations:0*
T0*
_output_shapes
: 2$
"MultivariateNormalDiagHead/truedivÔ
MultivariateNormalDiagHead/mulMul1MultivariateNormalDiagHead/Softplus:activations:0&MultivariateNormalDiagHead/truediv:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2 
MultivariateNormalDiagHead/mul
 MultivariateNormalDiagHead/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *˝752"
 MultivariateNormalDiagHead/add/yĘ
MultivariateNormalDiagHead/addAddV2"MultivariateNormalDiagHead/mul:z:0)MultivariateNormalDiagHead/add/y:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2 
MultivariateNormalDiagHead/addź
{MultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/range_dimension_tensor/ConstConst*
_output_shapes
: *
dtype0*
value	B :2}
{MultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/range_dimension_tensor/Constü
WMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/event_shapeConst*
_output_shapes
:*
dtype0*
valueB:2Y
WMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/event_shape
MultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/ShapeShape"MultivariateNormalDiagHead/add:z:0*
T0*
_output_shapes
:2
MultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/Shape¤
ĽMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙2¨
ĽMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack
§MultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2Ş
§MultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_1
§MultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2Ş
§MultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_2Ő
MultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_sliceStridedSlice MultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/Shape:output:0ŽMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack:output:0°MultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_1:output:0°MultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2˘
MultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_sliceŞ
ĄMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/values_1Pack¨MultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/strided_slice:output:0*
N*
T0*
_output_shapes
:2¤
ĄMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/values_1
MultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
MultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/axisę
MultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concatConcatV2 MultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/Shape:output:0ŞMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/values_1:output:0ŚMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat/axis:output:0*
N*
T0*
_output_shapes
:2
MultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concatŰ
MultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
MultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stackč
MultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ţ˙˙˙˙˙˙˙˙2
MultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_1ß
MultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
MultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_2˛
MultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_sliceStridedSliceĄMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/LinearOperatorDiag/shape_tensor/concat:output:0MultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack:output:0MultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_1:output:0MultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2
MultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice˙
QMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/ShapeShape)MultivariateNormalDiagHead/linear/Add:z:0*
T0*
_output_shapes
:2S
QMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/Shape
_MultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2a
_MultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack
aMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙2c
aMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_1
aMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2c
aMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_2¨
YMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_sliceStridedSliceZMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/Shape:output:0hMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack:output:0jMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_1:output:0jMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2[
YMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice
wMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/prefer_static_broadcast_shape/BroadcastArgsBroadcastArgsMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/LinearOperatorDiag/batch_shape_tensor/strided_slice:output:0bMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/strided_slice:output:0*
_output_shapes
:2y
wMultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/prefer_static_broadcast_shape/BroadcastArgsĂ
=MultivariateNormalDiagHead/MultivariateNormalDiag/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2?
=MultivariateNormalDiagHead/MultivariateNormalDiag/zeros/Constî
7MultivariateNormalDiagHead/MultivariateNormalDiag/zerosFill|MultivariateNormalDiagHead/MultivariateNormalDiag/shapes_from_loc_and_scale/prefer_static_broadcast_shape/BroadcastArgs:r0:0FMultivariateNormalDiagHead/MultivariateNormalDiag/zeros/Const:output:0*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙29
7MultivariateNormalDiagHead/MultivariateNormalDiag/zerosľ
6MultivariateNormalDiagHead/MultivariateNormalDiag/onesConst*
_output_shapes
: *
dtype0*
valueB
 *  ?28
6MultivariateNormalDiagHead/MultivariateNormalDiag/ones˛
6MultivariateNormalDiagHead/MultivariateNormalDiag/zeroConst*
_output_shapes
: *
dtype0*
value	B : 28
6MultivariateNormalDiagHead/MultivariateNormalDiag/zeroľ
7MultivariateNormalDiagHead/MultivariateNormalDiag/emptyConst*
_output_shapes
: *
dtype0*
valueB 29
7MultivariateNormalDiagHead/MultivariateNormalDiag/empty
^stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB 2`
^stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/sample_shape
âstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal/is_scalar_batch/is_scalar_batchConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2ĺ
âstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal/is_scalar_batch/is_scalar_batch
bstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/pick_vector/condConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2d
bstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/pick_vector/cond˘
jstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/pick_vector/false_vectorConst*
_output_shapes
:*
dtype0*
valueB:2l
jstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/pick_vector/false_vector
dstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/pick_vector_1/condConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2f
dstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/pick_vector_1/cond¤
kstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/pick_vector_1/true_vectorConst*
_output_shapes
:*
dtype0*
valueB:2m
kstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/pick_vector_1/true_vector
Ústochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/sample_shape/xConst*
_output_shapes
:*
dtype0*
valueB:2Ý
Ústochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/sample_shape/x
stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/sample_shape/xConst*
_output_shapes
:*
dtype0*
valueB"      2
stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/sample_shape/x
stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/ShapeShape@MultivariateNormalDiagHead/MultivariateNormalDiag/zeros:output:0*
T0*
_output_shapes
:2
stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/Shape
stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/BroadcastArgs/s1Const*
_output_shapes
: *
dtype0*
valueB 2
stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/BroadcastArgs/s1Ť	
stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/BroadcastArgsBroadcastArgsstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/Shape:output:0Ľstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/BroadcastArgs/s1:output:0*
_output_shapes
:2
stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/BroadcastArgs
stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:2
stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/concat/values_0÷
stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/concat/axisĐ
stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/concatConcatV2¤stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/concat/values_0:output:0stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/BroadcastArgs:r0:0 stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2
stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/concat
Ľstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2¨
Ľstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/normal/random_normal/mean
§stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2Ş
§stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/normal/random_normal/stddevć
ľstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormalstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/concat:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype02¸
ľstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/normal/random_normal/RandomStandardNormalü	
¤stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/normal/random_normal/mulMulžstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/normal/random_normal/RandomStandardNormal:output:0°stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/normal/random_normal/stddev:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2§
¤stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/normal/random_normal/mulÜ	
 stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/normal/random_normalAdd¨stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/normal/random_normal/mul:z:0Žstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/normal/random_normal/mean:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2Ł
 stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/normal/random_normalĆ
stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/mulMul¤stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/normal/random_normal:z:0?MultivariateNormalDiagHead/MultivariateNormalDiag/ones:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/mul¸
stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/addAddV2stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/mul:z:0@MultivariateNormalDiagHead/MultivariateNormalDiag/zeros:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/addń
stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/Shape_1Shapestochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/add:z:0*
T0*
_output_shapes
:2
stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/Shape_1
stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2˘
stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/strided_slice/stack
Ąstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2¤
Ąstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/strided_slice/stack_1
Ąstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2¤
Ąstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/strided_slice/stack_2Ż
stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/strided_sliceStridedSlicestochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/Shape_1:output:0¨stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/strided_slice/stack:output:0Şstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/strided_slice/stack_1:output:0Şstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2
stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/strided_sliceű
stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/concat_1/axisŮ
stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/concat_1ConcatV2Łstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/sample_shape/x:output:0˘stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/strided_slice:output:0˘stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/concat_1¤	
stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/ReshapeReshapestochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/add:z:0stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/concat_1:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙2
stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/Reshape
Ústochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2Ý
Ústochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/transpose/permů
Őstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/transpose	Transposestochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_Normal/sample/Reshape:output:0ăstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/transpose/perm:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙2Ř
Őstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/transposeł
Ństochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/ShapeShapeŮstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/transpose:y:0*
T0*
_output_shapes
:2Ô
Ństochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/Shape
ßstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2â
ßstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/strided_slice/stack
ástochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2ä
ástochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/strided_slice/stack_1
ástochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2ä
ástochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/strided_slice/stack_2­
Ůstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/strided_sliceStridedSliceÚstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/Shape:output:0čstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/strided_slice/stack:output:0ęstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/strided_slice/stack_1:output:0ęstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2Ü
Ůstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/strided_slice÷
×stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2Ú
×stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/concat/axis	
Ňstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/concatConcatV2ăstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/sample_shape/x:output:0âstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/strided_slice:output:0ŕstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2Ő
Ňstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/concat¨
Óstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/ReshapeReshapeŮstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/transpose:y:0Űstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/concat:output:0*
T0*+
_output_shapes
:˙˙˙˙˙˙˙˙˙2Ö
Óstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/Reshapeż
Wstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/ShapeShapeÜstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/Reshape:output:0*
T0*
_output_shapes
:2Y
Wstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/Shape
estochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2g
estochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/strided_slice/stack
gstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2i
gstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/strided_slice/stack_1
gstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2i
gstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/strided_slice/stack_2Ę
_stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/strided_sliceStridedSlice`stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/Shape:output:0nstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/strided_slice/stack:output:0pstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/strided_slice/stack_1:output:0pstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2a
_stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/strided_slice
]stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2_
]stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/concat/axisŠ
Xstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/concatConcatV2gstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/sample_shape:output:0hstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/strided_slice:output:0fstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2Z
Xstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/concatľ
Ystochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/ReshapeReshapeÜstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_SampleMultivariateNormalDiagHead_MultivariateNormalDiag_Normal_1/sample/Reshape:output:0astochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/concat:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2[
Ystochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/Reshapeî
stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_chain_of_MultivariateNormalDiagHead_MultivariateNormalDiag_shift_of_MultivariateNormalDiagHead_MultivariateNormalDiag_scale_matvec_linear_operator/forward/MultivariateNormalDiagHead_MultivariateNormalDiag_scale_matvec_linear_operator/forward/LinearOperatorDiag/matvec/mulMul"MultivariateNormalDiagHead/add:z:0bstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/Reshape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2
stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_chain_of_MultivariateNormalDiagHead_MultivariateNormalDiag_shift_of_MultivariateNormalDiagHead_MultivariateNormalDiag_scale_matvec_linear_operator/forward/MultivariateNormalDiagHead_MultivariateNormalDiag_scale_matvec_linear_operator/forward/LinearOperatorDiag/matvec/mulË	
âstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_chain_of_MultivariateNormalDiagHead_MultivariateNormalDiag_shift_of_MultivariateNormalDiagHead_MultivariateNormalDiag_scale_matvec_linear_operator/forward/MultivariateNormalDiagHead_MultivariateNormalDiag_shift/forward/addAddV2stochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_chain_of_MultivariateNormalDiagHead_MultivariateNormalDiag_shift_of_MultivariateNormalDiagHead_MultivariateNormalDiag_scale_matvec_linear_operator/forward/MultivariateNormalDiagHead_MultivariateNormalDiag_scale_matvec_linear_operator/forward/LinearOperatorDiag/matvec/mul:z:0)MultivariateNormalDiagHead/linear/Add:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2ĺ
âstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_chain_of_MultivariateNormalDiagHead_MultivariateNormalDiag_shift_of_MultivariateNormalDiagHead_MultivariateNormalDiag_scale_matvec_linear_operator/forward/MultivariateNormalDiagHead_MultivariateNormalDiag_shift/forward/addť
IdentityIdentityćstochastic_sampling_head/MultivariateNormalDiagHead_MultivariateNormalDiag/sample/MultivariateNormalDiagHead_MultivariateNormalDiag_chain_of_MultivariateNormalDiagHead_MultivariateNormalDiag_shift_of_MultivariateNormalDiagHead_MultivariateNormalDiag_scale_matvec_linear_operator/forward/MultivariateNormalDiagHead_MultivariateNormalDiag_shift/forward/add:z:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*Ă
_input_shapesą
Ž:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙	:::::::::::::::::::::::::::::::::::X T
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_nameargs_0/position:XT
'
_output_shapes
:˙˙˙˙˙˙˙˙˙	
)
_user_specified_nameargs_0/velocity:
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
: :#

_output_shapes
: " J
saver_filename:0StatefulPartitionedCall:0StatefulPartitionedCall_18"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp:Ý 
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
§
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
§
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
.:,Ź (2control_network/normalize/b
2:0	Ź (2control_network/normalize/w
4:2Ź (2!control_network/layer_norm/offset
3:1Ź (2 control_network/layer_norm/scale
9:7 (2&LayerNormAndResidualMLP/mlp/linear_0/b
>:<
Ź (2&LayerNormAndResidualMLP/mlp/linear_0/w
9:7 (2&LayerNormAndResidualMLP/mlp/linear_0/b
>:<
 (2&LayerNormAndResidualMLP/mlp/linear_0/w
9:7 (2&LayerNormAndResidualMLP/mlp/linear_1/b
>:<
 (2&LayerNormAndResidualMLP/mlp/linear_1/w
U:S (2BLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/offset
T:R (2ALayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/scale
9:7 (2&LayerNormAndResidualMLP/mlp/linear_0/b
>:<
 (2&LayerNormAndResidualMLP/mlp/linear_0/w
9:7 (2&LayerNormAndResidualMLP/mlp/linear_1/b
>:<
 (2&LayerNormAndResidualMLP/mlp/linear_1/w
U:S (2BLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/offset
T:R (2ALayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/scale
9:7 (2&LayerNormAndResidualMLP/mlp/linear_0/b
>:<
 (2&LayerNormAndResidualMLP/mlp/linear_0/w
9:7 (2&LayerNormAndResidualMLP/mlp/linear_1/b
>:<
 (2&LayerNormAndResidualMLP/mlp/linear_1/w
U:S (2BLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/offset
T:R (2ALayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/scale
9:7 (2&LayerNormAndResidualMLP/mlp/linear_0/b
>:<
 (2&LayerNormAndResidualMLP/mlp/linear_0/w
9:7 (2&LayerNormAndResidualMLP/mlp/linear_1/b
>:<
 (2&LayerNormAndResidualMLP/mlp/linear_1/w
U:S (2BLayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/offset
T:R (2ALayerNormAndResidualMLP/ResidualLayernormWrapper/layer_norm/scale
5:3 (2#MultivariateNormalDiagHead/linear/b
::8	 (2#MultivariateNormalDiagHead/linear/w
5:3 (2#MultivariateNormalDiagHead/linear/b
::8	 (2#MultivariateNormalDiagHead/linear/w
Ĺ2Â
__inference___call___248216˘
˛
FullArgSpec
args
jself
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Á2ž
!__inference_wrapped_module_248501
˛
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Ę2Ç
 __inference_initial_state_248504˘
˛
FullArgSpec
args
jself
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 ˙
__inference___call___248216ß"	
! #"%$˘
˘}
qŞn
5
position)&
args_0/position˙˙˙˙˙˙˙˙˙
5
velocity)&
args_0/velocity˙˙˙˙˙˙˙˙˙	
˘
˘ 
Ş "*˘'

0˙˙˙˙˙˙˙˙˙
˘
˘ N
 __inference_initial_state_248504*˘
˘

args_0 
Ş "˘
˘ 
!__inference_wrapped_module_248501ß"	
! #"%$˘
˘}
qŞn
5
position)&
args_0/position˙˙˙˙˙˙˙˙˙
5
velocity)&
args_0/velocity˙˙˙˙˙˙˙˙˙	
˘
˘ 
Ş "*˘'

0˙˙˙˙˙˙˙˙˙
˘
˘ 