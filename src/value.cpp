#include "value.h"

using PTR = std::shared_ptr<Value>;

Value::Value() : val{0}, grad{0}, parent1{nullptr}, parent2{nullptr}, op{None} {}
Value::Value(float val) : val{val}, grad{0}, parent1{nullptr}, parent2{nullptr}, op{None} {}
Value::Value(float val, std::shared_ptr<Value> &p1, Operation op): val{val}, parent1{p1}, parent2{nullptr}, op{op},parents {p1} {}

Value::Value(float val, std::shared_ptr<Value> &p1, std::shared_ptr<Value> &p2, Operation op):
  val{val},parent1{p1},parent2{p2},op{op}, parents{p1,p2}{}

std::shared_ptr<Value> operator+(std::shared_ptr<Value> &lhs, std::shared_ptr<Value> &rhs){
  return std::make_shared<Value>(lhs->val+rhs->val,lhs,rhs,Add);
}
std::shared_ptr<Value> operator*(std::shared_ptr<Value> &lhs, std::shared_ptr<Value> &rhs){
  return std::make_shared<Value>(lhs->val*rhs->val,lhs,rhs,Mul);
}


std::shared_ptr<Value> pow(std::shared_ptr<Value> &val, float other){
  //out = Value(self.data**other, (self,), f'**{other}')
  PTR ptr = std::make_shared<Value>(other);
  return std::make_shared<Value>(std::pow(val->val,other),val,ptr,Pow);
}

std::shared_ptr<Value> relu(std::shared_ptr<Value> &val){
  //out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')
  float num = val->val > 0 ? val->val : 0;
  return std::make_shared<Value>(num,val,Relu);
}

std::shared_ptr<Value> operator-(std::shared_ptr<Value> &val){
  PTR ptr = std::make_shared<Value>(-1);
  return val * ptr;
}

std::shared_ptr<Value> operator-(std::shared_ptr<Value> &lhs, std::shared_ptr<Value> &rhs){
  PTR ptr = -rhs;
  return lhs + ptr;
}

std::shared_ptr<Value> operator/(std::shared_ptr<Value> &lhs, std::shared_ptr<Value> &rhs){
  PTR ptr = pow(rhs,-1);
  return lhs * ptr;
}



void Value::back_add(){
  parent1->grad += grad;
  parent2->grad += grad;
}

void Value::back_mul(){
  parent1->grad += parent2->val* grad;
  parent2->grad += parent1->val* grad;
}

void Value::back_pow(){
  parent1->grad += (parent2->val * std::pow(parent1->val,parent2->val-1)) * grad;
}

void Value::back_relu(){
  float num = val > 0 ? 1 : 0;
  parent1->grad += num * grad;
}





void Value::get_grad_func(){
  switch(op) {
  case Add:
      back_add();
    break;
  case Sub:
    break;
  case Mul:
    back_mul();
    break;
  case Div:
    break;
  case Pow:
    back_pow();
    break;
  case Sig:
    break;
  case Relu:
    back_relu();
    break;
  default:
    break;
}
}

std::ostream& operator<<(std::ostream& os, const std::shared_ptr<Value>& obj){
  os << "VALUE: " << obj->val << " " << "GRADIENT: " << obj->grad << "\n";
  return os;
}


void build_topo(std::shared_ptr<Value> const &v, std::vector<std::shared_ptr<Value>> &topo, std::unordered_set<std::shared_ptr<Value>> &visited){
  if (!visited.count(v)){
    visited.insert(v);
    for (std::shared_ptr<Value> const &child : v->parents){
      build_topo(child, topo,visited);
    }
    topo.push_back(v);
  } 
}

void backward(std::shared_ptr<Value> &value){
  std::vector<std::shared_ptr<Value>> topo;
  std::unordered_set<std::shared_ptr<Value>> visited;
  build_topo(value, topo,visited);
  std::reverse(topo.begin(), topo.end());
  value->grad = 1;
  //std::cout << "THE SIZE OF TOPO IS: " << topo.size() << "\n";
  for (std::shared_ptr<Value> &v : topo){
    v->get_grad_func();
  }
}

//result[i][j] += A[i][k] * B[k][j];
//Plus Equals Multiply
std::shared_ptr<Value> PEM(std::shared_ptr<Value> &lhs, std::shared_ptr<Value> &middle, std::shared_ptr<Value> &rhs){
  //lhs->val *=2;
  PTR ptr = middle * rhs;
  return lhs + ptr;
}
