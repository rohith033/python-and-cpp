#include<bits/stdc++.h>
using namespace std;
/*encapsulation : putting to gether data members and functon in once such as class.
        data hideing (security purpose)
        fully encapsulation - making everything private.
        we can make our class read only without seting any set values 
        ~name destructor function
        */


//  class hero{
//            private:
//              string name;
//              int age;
//              public:
//              string get_name() { return "rohith"; }
//              int get_age() { return this->age;}

//        };
/*
inheratiance : 
        inherating from a parent class 
        class childname: type parent name{};
        multiple inheritance is also possible by adding commas in between the the parent classes
        hybrid mix of two or more inheritance types 
        inheritance ambugity name.::class 
*/
// class human{
//     public:
//     int age;
//      string name="rohith";
//      int weigth;
//      int get_age() {return this->age;}
//      int get_weigth() {return this->weigth;}
//      //string get_name() {return "rohith";}

// };
// class male: private human{
//     public: 
//     string color;
//     void eat(){
//         cout<<this->name+" eating"<<endl;
//     }
//   void get_name(){
//       cout<<this->name<<endl;
//   }
// };

/*
polymorpism : when a single things extists in multiple types it is said to be polymorph
compile type polymorphism :
           operator overloading
           returntype operator(op)(b &obj){
             datatype a = this->a;
             datatype b = obj.b;
             cout<<a(requried operation)b<<endl;
           }
           functional overloading
           function with differents types cant be overloaded it can be changed only in input args

run type polymorphism: 
     function overrdding
     inherittance has to done


/*
abstraction : information hideing
showing information hiding using acess modifiyers
*/


int main(){
  male rohith;
   rohith.get_name();
  //rohith.eat();
  //cout<<name<<endl;
}
