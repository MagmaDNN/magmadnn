#pragma once

#include <string>
#include <cstring>
#include <iostream>

namespace magmadnn {

class Arguments {
public:
   Arguments():
      enable_shortcut(true),
      learning_rate(0)
   {}   
   
   int parse(std::string const& context, int argc, char** argv) {
         
      int ret = 0;

      std::string help = 
         "Usage: " + context + " [options]\n" 
         R"(
Options:
--disable-shorcut     Disable shorcut in residual layers        
)";


         
      for( int i = 1; i < argc; ++i ) {

         if ( !strcmp("--help", argv[i]) ) {

            std::cout << help;
            return 1;
         }

         // Resnet
         else if ( !strcmp("--disable-shortcut", argv[i])) {
            enable_shortcut = false;
            std::cout << "Resnet: disable shortcuts"  << std::endl;
         }

         // SGD
         else if ( !strcmp("--learning-rate", argv[i]) && i+1 < argc ) {
            learning_rate =  std::stod( argv[++i] );
            std::cout << "SGD: Learning rate set to " << learning_rate << std::endl;
         }

      }

      return ret;
   }

public:

   // Resnet
   
   // Enable shorcut for residual layers. If set to `false`, simply
   // implements plain convolutional networks.
   bool enable_shortcut;

   // SGD
   double learning_rate;
};

} // End of namespace magmadnn
